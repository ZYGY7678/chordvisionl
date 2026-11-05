# העלאת ChordVision ל-Render (תוכנית חינמית)

כל הפרויקט (השרת, האתר ומנוע האקורדים) רץ כשירות אחד. אין צורך ב-Vercel.

## 1. להעלות ל-GitHub

1. ב-GitHub: **New repository**, למשל `chordvision` (אפשר פרטי).
2. **uploading an existing file**, ולגרור פנימה את **התוכן** של התיקייה `chordvision`. הקבצים `Dockerfile` ו-`render.yaml` חייבים להיות בשורש הריפו, לא בתוך תיקייה.
3. בפרויקט יש 98 קבצים, וההעלאה דרך הדפדפן מוגבלת ל-100 בבת אחת. אם משהו נעצר, מעלים קודם את התיקייה `frontend` ואחר כך את כל השאר.
4. לוודא שהקבצים שמתחילים בנקודה עלו: `.gitignore`, `.dockerignore`, `.env.example`.

## 2. לפרוס ב-Render

1. ב-Render: **New +** ← **Blueprint**, לחבר את חשבון ה-GitHub ולבחור את הריפו.
2. Render קורא את `render.yaml`, ומציג שירות אחד בשם `chordvision` בתוכנית **Free**. לוחצים **Apply**.
3. התוכנית חינמית, ולכן אין תשלום חודשי ואין דיסק.
4. הבנייה הראשונה לוקחת כמה דקות. כשהסטטוס **Live**, הכתובת מופיעה למעלה (`https://chordvision-xxxx.onrender.com`).
5. בדיקה: לפתוח `הכתובת/api/health`. אמורה להופיע תשובה עם `"status":"ok"`.

`CHORDVISION_SECRET_KEY` נוצר אוטומטית. **אל תשנה ואל תמחק אותו אחר כך**, אחרת כל המשתמשים מתנתקים ומפתחות ה-API השמורים נהיים בלתי קריאים.

## 3. מגבלות התוכנית החינמית

השרת מוגבל ל-512MB זיכרון ול-0.1 ליבה משותפת. מדדתי ניתוח בפועל על ליבה אחת מלאה:

| אורך השיר | זמן ניתוח (ליבה אחת מלאה) | זיכרון שיא של הניתוח |
|---|---|---|
| 4 דקות | כ-2.5 שניות | כ-340MB |
| 10 דקות | כ-6 שניות | כ-650MB |
| 20 דקות | כ-13 שניות | כ-1.2GB |

בתוכנית החינמית יש רק עשירית ליבה, כך שהזמנים יהיו בערך פי 10 מהטבלה (שיר של 4 דקות: כחצי דקה).

השרת עצמו תופס עוד כ-150MB. לכן ב-`render.yaml` מוגדר `CHORDVISION_MAX_AUDIO_SECONDS=300`, ושיר ארוך מ-5 דקות נדחה עם הודעה ברורה במקום להקריס את השרת.

## 4. דברים שחשוב לדעת על התוכנית החינמית

- **השרת נרדם אחרי 15 דקות בלי כניסות.** הכניסה הבאה לאתר מעירה אותו, וזה לוקח בערך דקה.
- **אין דיסק קבוע, והמידע נמחק.** פרויקטים והקלטות נשמרים רק כל עוד השרת ער. כשהוא נרדם, מופעל מחדש או מקבל דיפלוי, הכול מתאפס וכל מבקר מתחיל עם רשימת פרויקטים ריקה (האתר מתמודד עם זה בלי שגיאות). לכן כדאי לייצא (PDF, ChordPro וכו') מיד אחרי הניתוח.
- **שעות חינמיות**: Render נותן 750 שעות בחודש לחשבון, וזה מספיק לשירות אחד שרץ כל החודש.
- **דיפלוי או ריסטארט באמצע ניתוח**: הניתוח נעצר, והפרויקט מסומן "נכשל" עם כפתור לנסות שוב.
- **תמלול מילים** הוא אופציונלי: כל משתמש מכניס מפתח API משלו במסך ההגדרות. אפשר לבחור **Google Gemini (free)** ולהשתמש במפתח חינמי מ-[aistudio.google.com/apikey](https://aistudio.google.com/apikey). זיהוי האקורדים עצמו רץ כולו אצלך, ללא API חיצוני.
- **דומיין משלך**: **Settings** ← **Custom Domains**.
- **עדכונים**: כל `push` ל-GitHub מפעיל דיפלוי אוטומטי.

## 5. שדרוג בעתיד (אופציונלי)

אם בהמשך תרצה שהמידע יישמר ושהשרת לא יירדם:

1. ב-Render: **Settings** ← **Instance Type** ← **Starter** (512MB, כ-7$ לחודש) או **Standard** (2GB, כ-25$ לחודש, לשירים ארוכים).
2. **Disks** ← **Add Disk**, בשם `chordvision-data`, **Mount Path** `/data`.
3. לשירים ארוכים: **Environment** ← `CHORDVISION_MAX_AUDIO_SECONDS` ← `900` (רק ב-Standard).

## 6. אם משהו לא עובד

- הדיפלוי נכשל ← **Logs** בשירות, ולהעתיק לי את שורות השגיאה האחרונות.
- `/api/health` לא עונה ← אם השרת נרדם, לחכות דקה. אחרת לבדוק ב-Logs אם יש שגיאת `CHORDVISION_SECRET_KEY`.
- שיר ארוך נדחה ← זו המגבלה של 5 דקות (ראה סעיף 3).

## 7. הרצה מקומית (אופציונלי)

```bash
make install
./run.sh        # http://127.0.0.1:8000
make test       # כל הבדיקות
```


### Note about the replacement chord recognizer

The production worker now runs the supplied high-resolution CQT + NNLS
recognizer. Its extra librosa/Numba dependencies are included in the Docker
build. The recognizer is significantly more resource-intensive than the
previous implementation; Render Free's 512 MB memory limit may be insufficient
for some songs. The five-minute audio cap has been retained. If the service is
restarted during analysis, inspect the Render memory/log output before retrying.
