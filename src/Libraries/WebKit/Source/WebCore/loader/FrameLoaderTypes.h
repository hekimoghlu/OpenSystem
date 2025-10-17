/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 31, 2022.
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

#include "ElementContext.h"
#include "IntRect.h"
#include "ProcessIdentifier.h"

namespace WebCore {

enum class FrameState : uint8_t {
    Provisional,
    // This state indicates we are ready to commit to a page,
    // which means the view will transition to use the new data source.
    CommittedPage,
    Complete
};

enum class PolicyAction : uint8_t {
    Use,
    Download,
    Ignore,
    LoadWillContinueInAnotherProcess
};

enum class ReloadOption : uint8_t {
    ExpiredOnly = 1 << 0,
    FromOrigin  = 1 << 1,
    DisableContentBlockers = 1 << 2,
};

enum class FrameLoadType : uint8_t {
    Standard,
    Back,
    Forward,
    IndexedBackForward, // a multi-item hop in the backforward list
    Reload,
    Same, // user loads same URL again (but not reload button)
    RedirectWithLockedBackForwardList, // FIXME: Merge "lockBackForwardList", "lockHistory", "quickRedirect" and "clientRedirect" into a single concept of redirect.
    Replace,
    ReloadFromOrigin,
    ReloadExpiredOnly
};

enum class IsMetaRefresh : bool { No, Yes };
enum class WillContinueLoading : bool { No, Yes };
enum class WillInternallyHandleFailure : bool { No, Yes };

enum class ShouldContinuePolicyCheck : bool { No, Yes };

enum class NewFrameOpenerPolicy : uint8_t {
    Suppress,
    Allow
};

enum class NavigationType : uint8_t {
    LinkClicked,
    FormSubmitted,
    BackForward,
    Reload,
    FormResubmitted,
    Other
};

enum class NavigationHistoryBehavior : uint8_t {
    Auto,
    Push,
    Replace,
    Reload // Internal, not part of the specification
};

enum class ShouldOpenExternalURLsPolicy : uint8_t {
    ShouldNotAllow,
    ShouldAllowExternalSchemesButNotAppLinks,
    ShouldAllow,
};

enum class InitiatedByMainFrame : uint8_t {
    Yes,
    Unknown,
};

enum class ClearProvisionalItem : bool { No, Yes };

enum class StopLoadingPolicy {
    PreventDuringUnloadEvents,
    AlwaysStopLoading
};

enum class ObjectContentType : uint8_t {
    None,
    Image,
    Frame,
    PlugIn,
};

enum class UnloadEventPolicy {
    None,
    UnloadOnly,
    UnloadAndPageHide
};

enum class BrowsingContextGroupSwitchDecision : uint8_t {
    StayInGroup,
    NewSharedGroup,
    NewIsolatedGroup,
};

// Passed to FrameLoader::urlSelected() and ScriptController::executeIfJavaScriptURL()
// to control whether, in the case of a JavaScript URL, executeIfJavaScriptURL() should
// replace the document. It is a FIXME to eliminate this extra parameter from
// executeIfJavaScriptURL(), in which case this enum can go away.
enum ShouldReplaceDocumentIfJavaScriptURL {
    ReplaceDocumentIfJavaScriptURL,
    DoNotReplaceDocumentIfJavaScriptURL
};

enum class IsMainResourceLoad : bool { No, Yes };
enum class LockHistory : bool { No, Yes };
enum class LockBackForwardList : bool { No, Yes };
enum class AllowNavigationToInvalidURL : bool { No, Yes };
enum class HasInsecureContent : bool { No, Yes };
enum class LoadWillContinueInAnotherProcess : bool { No, Yes };

// FIXME: This should move to somewhere else. It no longer is related to frame loading.
struct SystemPreviewInfo {
    ElementContext element;

    IntRect previewRect;
    bool isPreview { false };
};

enum class LoadCompletionType : bool {
    Finish,
    Cancel
};

enum class AllowsContentJavaScript : bool {
    No,
    Yes,
};

enum class WindowProxyProperty : uint8_t {
    Other = 1 << 0,
    Closed = 1 << 1,
    PostMessage = 1 << 2,
};

} // namespace WebCore
