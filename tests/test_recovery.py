"""A restart must not leave projects stuck in "analysing" forever.

The job queue lives inside the API process, so a deploy or a crash drops
whatever was queued or running.  On the next start the server has to mark
those jobs failed, so the UI offers its retry button instead of showing a
progress bar that will never move.
"""
from __future__ import annotations

import os, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "engine"))
sys.path.insert(0, os.path.join(ROOT, "backend"))

WORK = tempfile.mkdtemp(prefix="chordvision-recovery-")
os.environ["CHORDVISION_STORAGE_DIR"] = os.path.join(WORK, "storage")
os.environ["CHORDVISION_DATABASE_URL"] = f"sqlite:///{os.path.join(WORK, 'test.db')}"
os.environ["CHORDVISION_SECRET_KEY"] = "test-secret-not-for-production"
os.environ["CHORDVISION_ENVIRONMENT"] = "development"

from app.db import (  # noqa: E402
    init_db, SessionLocal, User, Project, AnalysisJob, JobStatus, ProjectStatus,
)
from app.jobs import recover_interrupted_jobs  # noqa: E402

FAILED = []


def check(label, condition, detail=""):
    if condition:
        print(f"  PASS  {label}")
    else:
        FAILED.append(label)
        print(f"  FAIL  {label}{(': ' + detail) if detail else ''}")


def main() -> int:
    init_db()
    os.makedirs(os.environ["CHORDVISION_STORAGE_DIR"], exist_ok=True)
    s = SessionLocal()
    user = User(display_name=None)
    s.add(user)
    s.flush()

    def project(status):
        p = Project(user_id=user.id, title="t", file_name="t.mp3",
                    storage_key=f"k-{status}.mp3", content_type="audio/mpeg",
                    duration_ms=1000, file_size=1, status=status)
        s.add(p)
        s.flush()
        return p

    def job(p, status):
        j = AnalysisJob(project_id=p.id, status=status, stage="musical_analysis",
                        progress=0.4)
        s.add(j)
        s.flush()
        return j

    running_p = project(ProjectStatus.ANALYZING.value)
    running_j = job(running_p, JobStatus.RUNNING.value)
    queued_p = project(ProjectStatus.UPLOADED.value)
    queued_j = job(queued_p, JobStatus.QUEUED.value)
    done_p = project(ProjectStatus.READY.value)
    done_j = job(done_p, JobStatus.SUCCEEDED.value)
    s.commit()
    ids = (running_p.id, running_j.id, queued_p.id, queued_j.id, done_p.id, done_j.id)
    s.close()

    scratch = os.path.join(os.environ["CHORDVISION_STORAGE_DIR"], ".progress-abc")
    open(scratch, "w").write("x")

    print("\n[1] recovery after a restart")
    n = recover_interrupted_jobs()
    check("two interrupted jobs recovered", n == 2, str(n))

    s = SessionLocal()
    rp, rj, qp, qj, dp, dj = ids
    check("running job marked failed", s.get(AnalysisJob, rj).status == "failed")
    check("queued job marked failed", s.get(AnalysisJob, qj).status == "failed")
    check("failed job explains why",
          "restarted" in (s.get(AnalysisJob, rj).error_message or ""))
    check("analysing project marked failed", s.get(Project, rp).status == "failed")
    check("uploaded project marked failed", s.get(Project, qp).status == "failed")
    check("finished job untouched", s.get(AnalysisJob, dj).status == "succeeded")
    check("ready project untouched", s.get(Project, dp).status == "ready")
    check("progress scratch file removed", not os.path.exists(scratch))
    s.close()

    print("\n[2] running it again changes nothing")
    check("second run recovers nothing", recover_interrupted_jobs() == 0)

    print()
    if FAILED:
        print(f"{len(FAILED)} check(s) failed")
        return 1
    print("all recovery checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
