/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 27, 2022.
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
/*
 * os_qnx.c
 */

#include "vim.h"


#if defined(FEAT_GUI_PHOTON)
int is_photon_available;
#endif

void qnx_init(void)
{
#if defined(FEAT_GUI_PHOTON)
    PhChannelParms_t parms;

    CLEAR_FIELD(parms);
    parms.flags = Ph_DYNAMIC_BUFFER;

    is_photon_available = (PhAttach(NULL, &parms) != NULL) ? TRUE : FALSE;
#endif
}

#if (defined(FEAT_GUI_PHOTON) && defined(FEAT_CLIPBOARD)) || defined(PROTO)

#define CLIP_TYPE_VIM "VIMTYPE"
#define CLIP_TYPE_TEXT "TEXT"

// Turn on the clipboard for a console vim when photon is running
void qnx_clip_init(void)
{
    if (is_photon_available == TRUE && !gui.in_use)
	clip_init(TRUE);
}

/////////////////////////////////////////////////////////////////////////////
// Clipboard

// No support for owning the clipboard
int
clip_mch_own_selection(Clipboard_T *cbd)
{
    return FALSE;
}

void
clip_mch_lose_selection(Clipboard_T *cbd)
{
}

void
clip_mch_request_selection(Clipboard_T *cbd)
{
    int		    type = MLINE, clip_length = 0, is_type_set = FALSE;
    void	    *cbdata;
    PhClipHeader    *clip_header;
    char_u	    *clip_text = NULL;

    cbdata = PhClipboardPasteStart(PhInputGroup(NULL));
    if (cbdata == NULL)
	return;

    // Look for the vim specific clip first
    clip_header = PhClipboardPasteType(cbdata, CLIP_TYPE_VIM);
    if (clip_header != NULL && clip_header->data != NULL)
    {
	switch(*(char *) clip_header->data)
	{
	    default: // fallthrough to line type
	    case 'L': type = MLINE; break;
	    case 'C': type = MCHAR; break;
	    case 'B': type = MBLOCK; break;
	}
	is_type_set = TRUE;
    }

    // Try for just normal text
    clip_header = PhClipboardPasteType(cbdata, CLIP_TYPE_TEXT);
    if (clip_header != NULL)
    {
	clip_text = clip_header->data;
	clip_length  = clip_header->length - 1;

	if (clip_text != NULL && is_type_set == FALSE)
	    type = MAUTO;
    }

    if ((clip_text != NULL) && (clip_length > 0))
	clip_yank_selection(type, clip_text, clip_length, cbd);

    PhClipboardPasteFinish(cbdata);
}

void
clip_mch_set_selection(Clipboard_T *cbd)
{
    int type;
    long_u  len;
    char_u *text_clip, vim_clip[2], *str = NULL;
    PhClipHeader clip_header[2];

    // Prevent recursion from clip_get_selection()
    if (cbd->owned == TRUE)
	return;

    cbd->owned = TRUE;
    clip_get_selection(cbd);
    cbd->owned = FALSE;

    type = clip_convert_selection(&str, &len, cbd);
    if (type >= 0)
    {
	text_clip = alloc(len + 1); // Normal text

	if (text_clip && vim_clip)
	{
	    CLEAR_FIELD(clip_header);

	    STRNCPY(clip_header[0].type, CLIP_TYPE_VIM, 8);
	    clip_header[0].length = sizeof(vim_clip);
	    clip_header[0].data   = vim_clip;

	    STRNCPY(clip_header[1].type, CLIP_TYPE_TEXT, 8);
	    clip_header[1].length = len + 1;
	    clip_header[1].data   = text_clip;

	    switch(type)
	    {
		default: // fallthrough to MLINE
		case MLINE:	*vim_clip = 'L'; break;
		case MCHAR:	*vim_clip = 'C'; break;
		case MBLOCK:	*vim_clip = 'B'; break;
	    }

	    vim_strncpy(text_clip, str, len);

	    vim_clip[ 1 ] = NUL;

	    PhClipboardCopy(PhInputGroup(NULL), 2, clip_header);
	}
	vim_free(text_clip);
    }
    vim_free(str);
}
#endif
