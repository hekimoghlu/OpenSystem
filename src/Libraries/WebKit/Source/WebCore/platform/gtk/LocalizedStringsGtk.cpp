/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 22, 2024.
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
#include "config.h"
#include "LocalizedStrings.h"

#include <wtf/glib/GUniquePtr.h>
#include <wtf/text/CString.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

#if ENABLE(CONTEXT_MENUS)
String contextMenuItemTagCopyLinkToClipboard()
{
    return String::fromUTF8(_("Copy Link Loc_ation"));
}

String contextMenuItemTagDownloadImageToDisk()
{
    return String::fromUTF8(_("Sa_ve Image As"));
}

String contextMenuItemTagCopyImageURLToClipboard()
{
    return String::fromUTF8(_("Copy Image _Address"));
}

String contextMenuItemTagCopyVideoLinkToClipboard()
{
    return String::fromUTF8(_("Cop_y Video Link Location"));
}

String contextMenuItemTagCopyAudioLinkToClipboard()
{
    return String::fromUTF8(_("Cop_y Audio Link Location"));
}

String contextMenuItemTagToggleMediaControls()
{
    return String::fromUTF8(_("_Toggle Media Controls"));
}

String contextMenuItemTagShowMediaControls()
{
    return String::fromUTF8(_("_Show Media Controls"));
}

String contextMenuItemTagHideMediaControls()
{
    return String::fromUTF8(_("_Hide Media Controls"));
}

String contextMenuItemTagToggleMediaLoop()
{
    return String::fromUTF8(_("Toggle Media _Loop Playback"));
}

String contextMenuItemTagEnterVideoFullscreen()
{
    return String::fromUTF8(_("Switch Video to _Fullscreen"));
}

String contextMenuItemTagPasteAsPlainText()
{
    return String::fromUTF8(_("Paste As Plain _Text"));
}

String contextMenuItemTagDelete()
{
    return String::fromUTF8(_("_Delete"));
}

String contextMenuItemTagSelectAll()
{
    return String::fromUTF8(_("Select _All"));
}

String contextMenuItemTagInsertEmoji()
{
    return String::fromUTF8(_("Insert _Emoji"));
}

String contextMenuItemTagUnicode()
{
    return String::fromUTF8(_("_Insert Unicode Control Character"));
}

String contextMenuItemTagInputMethods()
{
    return String::fromUTF8(_("Input _Methods"));
}

String contextMenuItemTagUnicodeInsertLRMMark()
{
    return String::fromUTF8(_("LRM _Left-to-right mark"));
}

String contextMenuItemTagUnicodeInsertRLMMark()
{
    return String::fromUTF8(_("RLM _Right-to-left mark"));
}

String contextMenuItemTagUnicodeInsertLREMark()
{
    return String::fromUTF8(_("LRE Left-to-right _embedding"));
}

String contextMenuItemTagUnicodeInsertRLEMark()
{
    return String::fromUTF8(_("RLE Right-to-left e_mbedding"));
}

String contextMenuItemTagUnicodeInsertLROMark()
{
    return String::fromUTF8(_("LRO Left-to-right _override"));
}

String contextMenuItemTagUnicodeInsertRLOMark()
{
    return String::fromUTF8(_("RLO Right-to-left o_verride"));
}

String contextMenuItemTagUnicodeInsertPDFMark()
{
    return String::fromUTF8(_("PDF _Pop directional formatting"));
}

String contextMenuItemTagUnicodeInsertZWSMark()
{
    return String::fromUTF8(_("ZWS _Zero width space"));
}

String contextMenuItemTagUnicodeInsertZWJMark()
{
    return String::fromUTF8(_("ZWJ Zero width _joiner"));
}

String contextMenuItemTagUnicodeInsertZWNJMark()
{
    return String::fromUTF8(_("ZWNJ Zero width _non-joiner"));
}
#endif // ENABLE(CONTEXT_MENUS)

String validationMessageTooShortText(int, int minLength)
{
    GUniquePtr<char> string(g_strdup_printf(ngettext("Use at least one character", "Use at least %d characters", minLength), minLength));
    return String::fromUTF8(string.get());
}

String validationMessageTooLongText(int, int maxLength)
{
    GUniquePtr<char> string(g_strdup_printf(ngettext("Use no more than one character", "Use no more than %d characters", maxLength), maxLength));
    return String::fromUTF8(string.get());
}

} // namespace WebCore
