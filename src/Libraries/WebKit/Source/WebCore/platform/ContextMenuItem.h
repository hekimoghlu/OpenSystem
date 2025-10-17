/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 23, 2021.
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
#pragma once

#include <wtf/EnumTraits.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class ContextMenu;
class Image;

enum ContextMenuAction {
    ContextMenuItemTagNoAction,
    ContextMenuItemTagOpenLinkInNewWindow,
    ContextMenuItemTagDownloadLinkToDisk,
    ContextMenuItemTagCopyLinkToClipboard,
    ContextMenuItemTagOpenImageInNewWindow,
    ContextMenuItemTagDownloadImageToDisk,
    ContextMenuItemTagCopyImageToClipboard,
#if PLATFORM(GTK)
    ContextMenuItemTagCopyImageURLToClipboard,
#endif
    ContextMenuItemTagOpenFrameInNewWindow,
    ContextMenuItemTagCopy,
    ContextMenuItemTagGoBack,
    ContextMenuItemTagGoForward,
    ContextMenuItemTagStop,
    ContextMenuItemTagReload,
    ContextMenuItemTagCut,
    ContextMenuItemTagPaste,
#if PLATFORM(GTK)
    ContextMenuItemTagPasteAsPlainText,
    ContextMenuItemTagDelete,
    ContextMenuItemTagSelectAll,
    ContextMenuItemTagInputMethods,
    ContextMenuItemTagUnicode,
    ContextMenuItemTagUnicodeInsertLRMMark,
    ContextMenuItemTagUnicodeInsertRLMMark,
    ContextMenuItemTagUnicodeInsertLREMark,
    ContextMenuItemTagUnicodeInsertRLEMark,
    ContextMenuItemTagUnicodeInsertLROMark,
    ContextMenuItemTagUnicodeInsertRLOMark,
    ContextMenuItemTagUnicodeInsertPDFMark,
    ContextMenuItemTagUnicodeInsertZWSMark,
    ContextMenuItemTagUnicodeInsertZWJMark,
    ContextMenuItemTagUnicodeInsertZWNJMark,
    ContextMenuItemTagInsertEmoji,
#endif
    ContextMenuItemTagSpellingGuess,
    ContextMenuItemTagNoGuessesFound,
    ContextMenuItemTagIgnoreSpelling,
    ContextMenuItemTagLearnSpelling,
    ContextMenuItemTagOther,
#if PLATFORM(GTK)
    ContextMenuItemTagSearchWeb = 38,
#else
    ContextMenuItemTagSearchWeb = 21,
#endif
    ContextMenuItemTagLookUpInDictionary,
    ContextMenuItemTagOpenWithDefaultApplication,
    ContextMenuItemPDFActualSize,
    ContextMenuItemPDFZoomIn,
    ContextMenuItemPDFZoomOut,
    ContextMenuItemPDFAutoSize,
    ContextMenuItemPDFSinglePage,
    ContextMenuItemPDFFacingPages,
    ContextMenuItemPDFContinuous,
    ContextMenuItemPDFNextPage,
    ContextMenuItemPDFPreviousPage,
    ContextMenuItemTagOpenLink,
    ContextMenuItemTagIgnoreGrammar,
    ContextMenuItemTagSpellingMenu, // Spelling or Spelling/Grammar sub-menu
    ContextMenuItemTagShowSpellingPanel,
    ContextMenuItemTagCheckSpelling,
    ContextMenuItemTagCheckSpellingWhileTyping,
    ContextMenuItemTagCheckGrammarWithSpelling,
    ContextMenuItemTagFontMenu, // Font sub-menu
    ContextMenuItemTagShowFonts,
    ContextMenuItemTagBold,
    ContextMenuItemTagItalic,
    ContextMenuItemTagUnderline,
    ContextMenuItemTagOutline,
    ContextMenuItemTagStyles,
    ContextMenuItemTagShowColors,
    ContextMenuItemTagSpeechMenu, // Speech sub-menu
    ContextMenuItemTagStartSpeaking,
    ContextMenuItemTagStopSpeaking,
    ContextMenuItemTagWritingDirectionMenu, // Writing Direction sub-menu
    ContextMenuItemTagDefaultDirection,
    ContextMenuItemTagLeftToRight,
    ContextMenuItemTagRightToLeft,
    ContextMenuItemTagPDFSinglePageScrolling,
    ContextMenuItemTagPDFFacingPagesScrolling,
    ContextMenuItemTagInspectElement,
    ContextMenuItemTagTextDirectionMenu, // Text Direction sub-menu
    ContextMenuItemTagTextDirectionDefault,
    ContextMenuItemTagTextDirectionLeftToRight,
    ContextMenuItemTagTextDirectionRightToLeft,
#if PLATFORM(COCOA)
    ContextMenuItemTagCorrectSpellingAutomatically,
    ContextMenuItemTagSubstitutionsMenu,
    ContextMenuItemTagShowSubstitutions,
    ContextMenuItemTagSmartCopyPaste,
    ContextMenuItemTagSmartQuotes,
    ContextMenuItemTagSmartDashes,
    ContextMenuItemTagSmartLinks,
    ContextMenuItemTagTextReplacement,
    ContextMenuItemTagTransformationsMenu,
    ContextMenuItemTagMakeUpperCase,
    ContextMenuItemTagMakeLowerCase,
    ContextMenuItemTagCapitalize,
    ContextMenuItemTagChangeBack,
#endif
    ContextMenuItemTagOpenMediaInNewWindow,
    ContextMenuItemTagDownloadMediaToDisk,
    ContextMenuItemTagCopyMediaLinkToClipboard,
    ContextMenuItemTagToggleMediaControls,
    ContextMenuItemTagToggleMediaLoop,
    ContextMenuItemTagEnterVideoFullscreen,
    ContextMenuItemTagMediaPlayPause,
    ContextMenuItemTagMediaMute,
    ContextMenuItemTagDictationAlternative,
    ContextMenuItemTagPlayAllAnimations,
    ContextMenuItemTagPauseAllAnimations,
    ContextMenuItemTagPlayAnimation,
    ContextMenuItemTagPauseAnimation,
    ContextMenuItemTagToggleVideoFullscreen,
    ContextMenuItemTagShareMenu,
    ContextMenuItemTagToggleVideoEnhancedFullscreen,
    ContextMenuItemTagToggleVideoViewer,
    ContextMenuItemTagAddHighlightToCurrentQuickNote,
    ContextMenuItemTagAddHighlightToNewQuickNote,
    ContextMenuItemTagLookUpImage,
    ContextMenuItemTagTranslate,
    ContextMenuItemTagWritingTools,
    ContextMenuItemTagCopySubject,
    ContextMenuItemPDFSinglePageContinuous,
    ContextMenuItemPDFTwoPages,
    ContextMenuItemPDFTwoPagesContinuous,
    ContextMenuItemTagShowMediaStats,
    ContextMenuItemTagCopyLinkWithHighlight,
    ContextMenuItemLastNonCustomTag = ContextMenuItemTagCopyLinkWithHighlight,
    ContextMenuItemBaseCustomTag = 5000,
    ContextMenuItemLastCustomTag = 5999,
    ContextMenuItemBaseApplicationTag = 10000
};

enum class ContextMenuItemType : uint8_t {
    Action,
    CheckableAction,
    Separator,
    Submenu,
};

class ContextMenuItem {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(ContextMenuItem, WEBCORE_EXPORT);
public:
    WEBCORE_EXPORT ContextMenuItem(ContextMenuItemType, ContextMenuAction, const String&, ContextMenu* subMenu = 0);
    WEBCORE_EXPORT ContextMenuItem(ContextMenuItemType, ContextMenuAction, const String&, bool enabled, bool checked, unsigned indentationLevel = 0);

    WEBCORE_EXPORT ~ContextMenuItem();

    void setType(ContextMenuItemType);
    WEBCORE_EXPORT ContextMenuItemType type() const;

    void setAction(ContextMenuAction);
    WEBCORE_EXPORT ContextMenuAction action() const;

    void setChecked(bool = true);
    WEBCORE_EXPORT bool checked() const;

    void setEnabled(bool = true);
    WEBCORE_EXPORT bool enabled() const;

    void setIndentationLevel(unsigned);
    WEBCORE_EXPORT unsigned indentationLevel() const;

    void setSubMenu(ContextMenu*);

    WEBCORE_EXPORT ContextMenuItem(ContextMenuAction, const String&, bool enabled, bool checked, const Vector<ContextMenuItem>& subMenuItems, unsigned indentationLevel = 0);
    ContextMenuItem();

    bool isNull() const;

    void setTitle(const String& title) { m_title = title; }
    const String& title() const { return m_title; }

    const Vector<ContextMenuItem>& subMenuItems() const { return m_subMenuItems; }
private:
    ContextMenuItemType m_type;
    ContextMenuAction m_action;
    String m_title;
    bool m_enabled;
    bool m_checked;
    unsigned m_indentationLevel;
    Vector<ContextMenuItem> m_subMenuItems;
};

} // namespace WebCore

namespace WTF {

template<> WEBCORE_EXPORT bool isValidEnum<WebCore::ContextMenuAction>(std::underlying_type_t<WebCore::ContextMenuAction>);

} // namespace WTF
