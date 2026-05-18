import sys
sys.path.insert(0, '.')
from src.ingestion import _fix_char_spacing, _clean_pdf_text

# Simulate the garbled text from the user's PDF
garbled = (
    "i n t ( b i s g r e a t e r than a )\n"
    "8 e l i f a == b :\n"
    "9 p r i n t ( a i s e q u a l t o b )\n"
    "5 . 2 A r r a y s\n"
    "A n a r r a y i s a s p e c i a l v a r i a b l e , w h i c h c a n h o l d m o r e t h a n o n e v a l u e a t a t i m e .\n"
    "H e r e a r e s o m e E x a m p l e s h o w y o u c a n c r e a t e a n d u s e A r r a y s i n P y t h o n :\n"
    "d a t a = [ 1 . 6 , 3 . 4 , 5 . 5 , 9 . 4 ]\n"
    "f o r x i n d a t a :\n"
    "    p r i n t ( x )\n"
    "c a r l i s t = [ 'T o y o t a' , 'V o l v o' , 'B M W' ]\n"
    "f o r x i n c a r l i s t :\n"
    "    p r i n t ( x )\n"
)

cleaned = _clean_pdf_text(garbled)
print("=== CLEANED OUTPUT ===")
print(cleaned)
print()
print(f"Before: {len(garbled)} chars | After: {len(cleaned)} chars")
