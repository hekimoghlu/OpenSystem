/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 25, 2022.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
/****************************************************************************
 *   Author:  Juergen Pfeifer, 1995,1997                                    *
 ****************************************************************************/

#include "form.priv.h"

MODULE_ID("$Id: frm_data.c,v 1.16 2013/08/24 22:44:05 tom Exp $")

/*---------------------------------------------------------------------------
|   Facility      :  libnform  
|   Function      :  bool data_behind(const FORM *form)
|   
|   Description   :  Check for off-screen data behind. This is nearly trivial
|                    because the beginning of a field is fixed.
|
|   Return Values :  TRUE   - there are off-screen data behind
|                    FALSE  - there are no off-screen data behind
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(bool)
data_behind(const FORM *form)
{
  bool result = FALSE;

  T((T_CALLED("data_behind(%p)"), (const void *)form));

  if (form && (form->status & _POSTED) && form->current)
    {
      FIELD *field;

      field = form->current;
      if (!Single_Line_Field(field))
	{
	  result = (form->toprow == 0) ? FALSE : TRUE;
	}
      else
	{
	  result = (form->begincol == 0) ? FALSE : TRUE;
	}
    }
  returnBool(result);
}

/*---------------------------------------------------------------------------
|   Facility      :  libnform  
|   Function      :  static char * Only_Padding(
|                                    WINDOW *w,
|                                    int len,
|                                    int pad)
|   
|   Description   :  Test if 'length' cells starting at the current position
|                    contain a padding character.
|
|   Return Values :  true if only padding cells are found
+--------------------------------------------------------------------------*/
NCURSES_INLINE static bool
Only_Padding(WINDOW *w, int len, int pad)
{
  bool result = TRUE;
  int y, x, j;
  FIELD_CELL cell;

  getyx(w, y, x);
  for (j = 0; j < len; ++j)
    {
      if (wmove(w, y, x + j) != ERR)
	{
#if USE_WIDEC_SUPPORT
	  if (win_wch(w, &cell) != ERR)
	    {
	      if ((chtype)CharOf(cell) != ChCharOf(pad)
		  || cell.chars[1] != 0)
		{
		  result = FALSE;
		  break;
		}
	    }
#else
	  cell = (FIELD_CELL) winch(w);
	  if (ChCharOf(cell) != ChCharOf(pad))
	    {
	      result = FALSE;
	      break;
	    }
#endif
	}
      else
	{
	  /* if an error, return true: no non-padding text found */
	  break;
	}
    }
  /* no need to reset the cursor position; caller does this */
  return result;
}

/*---------------------------------------------------------------------------
|   Facility      :  libnform  
|   Function      :  bool data_ahead(const FORM *form)
|   
|   Description   :  Check for off-screen data ahead. This is more difficult
|                    because a dynamic field has a variable end. 
|
|   Return Values :  TRUE   - there are off-screen data ahead
|                    FALSE  - there are no off-screen data ahead
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(bool)
data_ahead(const FORM *form)
{
  bool result = FALSE;

  T((T_CALLED("data_ahead(%p)"), (const void *)form));

  if (form && (form->status & _POSTED) && form->current)
    {
      FIELD *field;
      bool cursor_moved = FALSE;
      int pos;

      field = form->current;
      assert(form->w);

      if (Single_Line_Field(field))
	{
	  int check_len;

	  pos = form->begincol + field->cols;
	  while (pos < field->dcols)
	    {
	      check_len = field->dcols - pos;
	      if (check_len >= field->cols)
		check_len = field->cols;
	      cursor_moved = TRUE;
	      wmove(form->w, 0, pos);
	      if (Only_Padding(form->w, check_len, field->pad))
		pos += field->cols;
	      else
		{
		  result = TRUE;
		  break;
		}
	    }
	}
      else
	{
	  pos = form->toprow + field->rows;
	  while (pos < field->drows)
	    {
	      cursor_moved = TRUE;
	      wmove(form->w, pos, 0);
	      pos++;
	      if (!Only_Padding(form->w, field->cols, field->pad))
		{
		  result = TRUE;
		  break;
		}
	    }
	}

      if (cursor_moved)
	wmove(form->w, form->currow, form->curcol);
    }
  returnBool(result);
}

/* frm_data.c ends here */
