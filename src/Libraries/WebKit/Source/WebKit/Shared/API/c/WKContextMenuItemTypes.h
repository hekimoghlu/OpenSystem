/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 15, 2023.
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
#ifndef WKContextMenuItemTypes_h
#define WKContextMenuItemTypes_h

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    kWKContextMenuItemTagNoAction = 0,
    kWKContextMenuItemTagOpenLinkInNewWindow,
    kWKContextMenuItemTagDownloadLinkToDisk,
    kWKContextMenuItemTagCopyLinkToClipboard,
    kWKContextMenuItemTagOpenImageInNewWindow,
    kWKContextMenuItemTagDownloadImageToDisk,
    kWKContextMenuItemTagCopyImageToClipboard,
    kWKContextMenuItemTagOpenFrameInNewWindow,
    kWKContextMenuItemTagCopy,
    kWKContextMenuItemTagGoBack,
    kWKContextMenuItemTagGoForward,
    kWKContextMenuItemTagStop,
    kWKContextMenuItemTagReload,
    kWKContextMenuItemTagCut,
    kWKContextMenuItemTagPaste,
    kWKContextMenuItemTagSpellingGuess,
    kWKContextMenuItemTagNoGuessesFound,
    kWKContextMenuItemTagIgnoreSpelling,
    kWKContextMenuItemTagLearnSpelling,
    kWKContextMenuItemTagOther,
    kWKContextMenuItemTagSearchInSpotlight,
    kWKContextMenuItemTagSearchWeb,
    kWKContextMenuItemTagLookUpInDictionary,
    kWKContextMenuItemTagOpenWithDefaultApplication,
    kWKContextMenuItemTagPDFActualSize,
    kWKContextMenuItemTagPDFZoomIn,
    kWKContextMenuItemTagPDFZoomOut,
    kWKContextMenuItemTagPDFAutoSize,
    kWKContextMenuItemTagPDFSinglePage,
    kWKContextMenuItemTagPDFFacingPages,
    kWKContextMenuItemTagPDFContinuous,
    kWKContextMenuItemTagPDFNextPage,
    kWKContextMenuItemTagPDFPreviousPage,
    kWKContextMenuItemTagOpenLink,
    kWKContextMenuItemTagIgnoreGrammar,
    kWKContextMenuItemTagSpellingMenu, 
    kWKContextMenuItemTagShowSpellingPanel,
    kWKContextMenuItemTagCheckSpelling,
    kWKContextMenuItemTagCheckSpellingWhileTyping,
    kWKContextMenuItemTagCheckGrammarWithSpelling,
    kWKContextMenuItemTagFontMenu, 
    kWKContextMenuItemTagShowFonts,
    kWKContextMenuItemTagBold,
    kWKContextMenuItemTagItalic,
    kWKContextMenuItemTagUnderline,
    kWKContextMenuItemTagOutline,
    kWKContextMenuItemTagStyles,
    kWKContextMenuItemTagShowColors,
    kWKContextMenuItemTagSpeechMenu, 
    kWKContextMenuItemTagStartSpeaking,
    kWKContextMenuItemTagStopSpeaking,
    kWKContextMenuItemTagWritingDirectionMenu, 
    kWKContextMenuItemTagDefaultDirection,
    kWKContextMenuItemTagLeftToRight,
    kWKContextMenuItemTagRightToLeft,
    kWKContextMenuItemTagPDFSinglePageScrolling,
    kWKContextMenuItemTagPDFFacingPagesScrolling,
    kWKContextMenuItemTagInspectElement,
    kWKContextMenuItemTagTextDirectionMenu,
    kWKContextMenuItemTagTextDirectionDefault,
    kWKContextMenuItemTagTextDirectionLeftToRight,
    kWKContextMenuItemTagTextDirectionRightToLeft,
    kWKContextMenuItemTagCorrectSpellingAutomatically,
    kWKContextMenuItemTagSubstitutionsMenu,
    kWKContextMenuItemTagShowSubstitutions,
    kWKContextMenuItemTagSmartCopyPaste,
    kWKContextMenuItemTagSmartQuotes,
    kWKContextMenuItemTagSmartDashes,
    kWKContextMenuItemTagSmartLinks,
    kWKContextMenuItemTagTextReplacement,
    kWKContextMenuItemTagTransformationsMenu,
    kWKContextMenuItemTagMakeUpperCase,
    kWKContextMenuItemTagMakeLowerCase,
    kWKContextMenuItemTagCapitalize,
    kWKContextMenuItemTagChangeBack,
    kWKContextMenuItemTagOpenMediaInNewWindow,
    kWKContextMenuItemTagDownloadMediaToDisk,
    kWKContextMenuItemTagCopyMediaLinkToClipboard,
    kWKContextMenuItemTagToggleMediaControls,
    kWKContextMenuItemTagToggleMediaLoop,
    kWKContextMenuItemTagEnterVideoFullscreen,
    kWKContextMenuItemTagMediaPlayPause,
    kWKContextMenuItemTagMediaMute,
    kWKContextMenuItemTagDictationAlternative,
    kWKContextMenuItemTagPlayAllAnimations,
    kWKContextMenuItemTagPauseAllAnimations,
    kWKContextMenuItemTagPlayAnimation,
    kWKContextMenuItemTagPauseAnimation,
    kWKContextMenuItemTagCopyImageURLToClipboard,
    kWKContextMenuItemTagSelectAll,
    kWKContextMenuItemTagOpenLinkInThisWindow,
    kWKContextMenuItemTagToggleVideoFullscreen,
    kWKContextMenuItemTagShareMenu,
    kWKContextMenuItemTagToggleVideoEnhancedFullscreen,
    kWKContextMenuItemTagToggleVideoViewer,
    kWKContextMenuItemTagAddHighlightToCurrentQuickNote,
    kWKContextMenuItemTagAddHighlightToNewQuickNote,
    kWKContextMenuItemTagRevealImage,
    kWKContextMenuItemTagTranslate,
    kWKContextMenuItemTagCopyCroppedImage,
    kWKContextMenuItemTagWritingTools,
    kWKContextMenuItemTagCopyLinkWithHighlight,
    kWKContextMenuItemBaseApplicationTag = 10000
};
typedef uint32_t WKContextMenuItemTag;

enum {
    kWKContextMenuItemTypeAction,
    kWKContextMenuItemTypeCheckableAction,
    kWKContextMenuItemTypeSeparator,
    kWKContextMenuItemTypeSubmenu
};
typedef uint32_t WKContextMenuItemType;
    
#ifdef __cplusplus
}
#endif

#endif /* WKContextMenuItemTypes_h */
