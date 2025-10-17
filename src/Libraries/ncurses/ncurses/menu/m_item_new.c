/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 17, 2021.
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

/***************************************************************************
* Module m_item_new                                                        *
* Create and destroy menu items                                            *
* Set and get marker string for menu                                       *
***************************************************************************/

#include "menu.priv.h"

#if USE_WIDEC_SUPPORT
#if HAVE_WCTYPE_H
#include <wctype.h>
#endif
#endif

MODULE_ID("$Id: m_item_new.c,v 1.33 2012/06/09 23:55:15 tom Exp $")

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  bool Is_Printable_String(const char *s)
|   
|   Description   :  Checks whether or not the string contains only printable
|                    characters.
|
|   Return Values :  TRUE     - if string is printable
|                    FALSE    - if string contains non-printable characters
+--------------------------------------------------------------------------*/
static bool
Is_Printable_String(const char *s)
{
  int result = TRUE;

#if USE_WIDEC_SUPPORT
  int count = (int)mbstowcs(0, s, 0);
  wchar_t *temp = 0;

  assert(s);

  if (count > 0
      && (temp = typeCalloc(wchar_t, (2 + (unsigned)count))) != 0)
    {
      int n;

      mbstowcs(temp, s, (unsigned)count);
      for (n = 0; n < count; ++n)
	if (!iswprint((wint_t) temp[n]))
	  {
	    result = FALSE;
	    break;
	  }
      free(temp);
    }
#else
  assert(s);
  while (*s)
    {
      if (!isprint(UChar(*s)))
	{
	  result = FALSE;
	  break;
	}
      s++;
    }
#endif
  return result;
}

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  ITEM *new_item(char *name, char *description)
|   
|   Description   :  Create a new item with name and description. Return
|                    a pointer to this new item.
|                    N.B.: an item must(!) have a name.
|
|   Return Values :  The item pointer or NULL if creation failed.
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(ITEM *)
new_item(const char *name, const char *description)
{
  ITEM *item;

  T((T_CALLED("new_item(\"%s\", \"%s\")"),
     name ? name : "",
     description ? description : ""));

  if (!name || (*name == '\0') || !Is_Printable_String(name))
    {
      item = (ITEM *) 0;
      SET_ERROR(E_BAD_ARGUMENT);
    }
  else
    {
      item = typeCalloc(ITEM, 1);
      if (item)
	{
	  *item = _nc_Default_Item;	/* hope we have struct assignment */

	  item->name.length = (unsigned short)strlen(name);
	  item->name.str = name;

	  if (description && (*description != '\0') &&
	      Is_Printable_String(description))
	    {
	      item->description.length = (unsigned short)strlen(description);
	      item->description.str = description;
	    }
	  else
	    {
	      item->description.length = 0;
	      item->description.str = (char *)0;
	    }
	}
      else
	SET_ERROR(E_SYSTEM_ERROR);
    }
  returnItem(item);
}

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  int free_item(ITEM *item)
|   
|   Description   :  Free the allocated storage for this item. 
|                    N.B.: a connected item can't be freed.
|
|   Return Values :  E_OK              - success
|                    E_BAD_ARGUMENT    - invalid value has been passed
|                    E_CONNECTED       - item is still connected to a menu    
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
free_item(ITEM * item)
{
  T((T_CALLED("free_item(%p)"), (void *)item));

  if (!item)
    RETURN(E_BAD_ARGUMENT);

  if (item->imenu)
    RETURN(E_CONNECTED);

  free(item);

  RETURN(E_OK);
}

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  int set_menu_mark( MENU *menu, const char *mark )
|   
|   Description   :  Set the mark string used to indicate the current
|                    item (single-valued menu) or the selected items
|                    (multi-valued menu).
|                    The mark argument may be NULL, in which case no 
|                    marker is used.
|                    This might be a little bit tricky, because this may 
|                    affect the geometry of the menu, which we don't allow 
|                    if it is already posted.
|
|   Return Values :  E_OK               - success
|                    E_BAD_ARGUMENT     - an invalid value has been passed
|                    E_SYSTEM_ERROR     - no memory to store mark
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(int)
set_menu_mark(MENU * menu, const char *mark)
{
  short l;

  T((T_CALLED("set_menu_mark(%p,%s)"), (void *)menu, _nc_visbuf(mark)));

  if (mark && (*mark != '\0') && Is_Printable_String(mark))
    l = (short)strlen(mark);
  else
    l = 0;

  if (menu)
    {
      char *old_mark = menu->mark;
      unsigned short old_status = menu->status;

      if (menu->status & _POSTED)
	{
	  /* If the menu is already posted, the geometry is fixed. Then
	     we can only accept a mark with exactly the same length */
	  if (menu->marklen != l)
	    RETURN(E_BAD_ARGUMENT);
	}
      menu->marklen = l;
      if (l)
	{
	  menu->mark = strdup(mark);
	  if (menu->mark)
	    {
	      if (menu != &_nc_Default_Menu)
		SetStatus(menu, _MARK_ALLOCATED);
	    }
	  else
	    {
	      menu->mark = old_mark;
	      menu->marklen = (short)((old_mark != 0) ? strlen(old_mark) : 0);
	      RETURN(E_SYSTEM_ERROR);
	    }
	}
      else
	menu->mark = (char *)0;

      if ((old_status & _MARK_ALLOCATED) && old_mark)
	free(old_mark);

      if (menu->status & _POSTED)
	{
	  _nc_Draw_Menu(menu);
	  _nc_Show_Menu(menu);
	}
      else
	{
	  /* Recalculate the geometry */
	  _nc_Calculate_Item_Length_and_Width(menu);
	}
    }
  else
    {
      returnCode(set_menu_mark(&_nc_Default_Menu, mark));
    }
  RETURN(E_OK);
}

/*---------------------------------------------------------------------------
|   Facility      :  libnmenu  
|   Function      :  char *menu_mark(const MENU *menu)
|   
|   Description   :  Return a pointer to the marker string
|
|   Return Values :  The marker string pointer or NULL if no marker defined
+--------------------------------------------------------------------------*/
NCURSES_EXPORT(const char *)
menu_mark(const MENU * menu)
{
  T((T_CALLED("menu_mark(%p)"), (const void *)menu));
  returnPtr(Normalize_Menu(menu)->mark);
}

/* m_item_new.c */
