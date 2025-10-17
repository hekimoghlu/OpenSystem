/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 2, 2022.
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

#if PLATFORM(COCOA)

#include <optional>
#include <wtf/BitSet.h>
#include <wtf/Forward.h>
#include <wtf/RuntimeApplicationChecks.h>

namespace WTF {

enum class SDKAlignedBehavior {
    AllowsWheelEventGesturesToBecomeNonBlocking,
    ApplicationStateTrackerDoesNotObserveWindow,
    AuthorizationHeaderOnSameOriginRedirects,
    BlanksViewOnJSPrompt,
    BlocksConnectionsToAddressWithOnlyZeros,
    BrowsingContextControllerSPIAccessRemoved,
    ContextMenuTriggersLinkActivationNavigationType,
    ConvertsInvalidURLsToBlank,
    DataURLFragmentRemoval,
    DecidesPolicyBeforeLoadingQuickLookPreview,
    DefaultsToExcludingBackgroundsWhenPrinting,
    DefaultsToPassiveTouchListenersOnDocument,
    DefaultsToPassiveWheelListenersOnDocument,
    DisallowsSettingAnyXHRHeaderFromFileURLs,
    DoesNotDrainTheMicrotaskQueueWhenCallingObjC,
    DoesNotParseStringEndingWithFullStopAsFloatingPointNumber,
    DoesNotAddIntrinsicMarginsToFormControls,
    DOMWindowReuseRestriction,
    DownloadDelegatesCalledOnTheMainThread,
    DropToNavigateDisallowedByDefault,
    ExceptionsForDuplicateCompletionHandlerCalls,
    ExceptionsForRelatedWebViewsUsingDifferentDataStores,
    ExpiredOnlyReloadBehavior,
    ForbidsDotPrefixedFonts,
    FullySuspendsBackgroundContent,
    FullySuspendsBackgroundContentImmediately,
    HasUIContextMenuInteraction,
    HTMLDocumentSupportedPropertyNames,
    InitializeWebKit2MainThreadAssertion,
    InspectableDefaultsToDisabled,
    LazyGestureRecognizerInstallation,
    LinkPreviewEnabledByDefault,
    MainThreadReleaseAssertionInWebPageProxy,
    MediaTypesRequiringUserActionForPlayback,
    MinimizesLanguages,
    ModernCompabilityModeByDefault,
    NoClientCertificateLookup,
    NoExpandoIndexedPropertiesOnWindow,
    NoGetElementsByNameQuirk,
    NoIMDbCSSOMViewScrollingQuirk,
    NoLaBanquePostaleQuirks,
    NoMoviStarPlusCORSPreflightQuirk,
    NoPokerBrosBuiltInTagQuirk,
    NoShowModalDialog,
    NoTheSecretSocietyHiddenMysteryWindowOpenQuirk,
    NoTypedArrayAPIQuirk,
    NoUnconditionalUniversalSandboxExtension,
    NoWeChatScrollingQuirk,
    NoUNIQLOLazyIframeLoadingQuirk,
    NullOriginForNonSpecialSchemedURLs,
    ObservesClassProperty,
    PictureInPictureMediaPlayback,
    ProcessSwapOnCrossSiteNavigation,
    PushStateFilePathRestriction,
    RequiresUserGestureToLoadVideo,
    RestrictsBaseURLSchemes,
    RunningBoardThrottling,
    ScrollViewContentInsetsAreNotObscuringInsets,
    SendsNativeMouseEvents,
    SessionCleanupByDefault,
    SharedNetworkProcess,
    SiteSpecificQuirksAreEnabledByDefault,
    SnapshotAfterScreenUpdates,
    SupportsDeviceOrientationAndMotionPermissionAPI,
    SupportsInitConstructors,
    SupportsiOSAppsOnMacOS,
    SupportsOverflowHiddenOnMainFrame,
    TimerThreadSafetyChecks,
    UIScrollViewDoesNotApplyKeyboardInsetsUnconditionally,
    UnprefixedPlaysInlineAttribute,
    WebIconDatabaseWarning,
    WebSQLDisabledByDefaultInLegacyWebKit,
    WKContentViewDoesNotOverrideKeyCommands,
    WKWebsiteDataStoreInitReturningNil,
    UIBackForwardSkipsHistoryItemsWithoutUserGesture,
    ProgrammaticFocusDuringUserScriptShowsInputViews,
    UsesGameControllerPhysicalInputProfile,
    ScreenOrientationAPIEnabled,
    PopoverAttributeEnabled,
    LiveRangeSelectionEnabledForAllApps,
    DoesNotOverrideUAFromNSUserDefault,
    EvaluateJavaScriptWithoutTransientActivation,
    ResettingTransitionCancelsRunningTransitionQuirk,
    OnlyLoadWellKnownAboutURLs,
    AsyncFragmentNavigationPolicyDecision,
    DoNotLoadStyleSheetIfHTTPStatusIsNotOK,
    ScrollViewSubclassImplementsAddGestureRecognizer,
    ThrowIfCanDeclareGlobalFunctionFails,
    ThrowOnKVCInstanceVariableAccess,
    LaxCookieSameSiteAttribute,
    BlockOptionallyBlockableMixedContent,
    UseCFNetworkNetworkLoader,
    BlockCrossOriginRedirectDownloads,
    BlobFileAccessEnforcementAndNetworkProcessRoundTrip,
    DevolvableWidgets,
    SetSelectionRangeCachesSelectionIfNotFocusedOrSelected,
    DispatchFocusEventBeforeNotifyingClient,
    BlobFileAccessEnforcement,

    NumberOfBehaviors
};

using SDKAlignedBehaviors = WTF::BitSet<static_cast<size_t>(SDKAlignedBehavior::NumberOfBehaviors), uint32_t>;

WTF_EXPORT_PRIVATE const SDKAlignedBehaviors& sdkAlignedBehaviors();
WTF_EXPORT_PRIVATE void setSDKAlignedBehaviors(SDKAlignedBehaviors);

WTF_EXPORT_PRIVATE void enableAllSDKAlignedBehaviors();
WTF_EXPORT_PRIVATE void disableAllSDKAlignedBehaviors();

WTF_EXPORT_PRIVATE bool linkedOnOrAfterSDKWithBehavior(SDKAlignedBehavior);

WTF_EXPORT_PRIVATE bool processIsExtension();
WTF_EXPORT_PRIVATE void setProcessIsExtension(bool);

WTF_EXPORT_PRIVATE void setApplicationBundleIdentifier(const String&);
WTF_EXPORT_PRIVATE void setApplicationBundleIdentifierOverride(const String&);
WTF_EXPORT_PRIVATE String applicationBundleIdentifier();
WTF_EXPORT_PRIVATE void clearApplicationBundleIdentifierTestingOverride();

namespace CocoaApplication {

WTF_EXPORT_PRIVATE bool isIBooks();
WTF_EXPORT_PRIVATE bool isWebkitTestRunner();

}

#if PLATFORM(MAC)

namespace MacApplication {

WTF_EXPORT_PRIVATE bool isAdobeInstaller();
WTF_EXPORT_PRIVATE bool isAppleMail();
WTF_EXPORT_PRIVATE bool isMiniBrowser();
WTF_EXPORT_PRIVATE bool isQuickenEssentials();
WTF_EXPORT_PRIVATE bool isSafari();
WTF_EXPORT_PRIVATE bool isVersions();
WTF_EXPORT_PRIVATE bool isHRBlock();
WTF_EXPORT_PRIVATE bool isEpsonSoftwareUpdater();
WTF_EXPORT_PRIVATE bool isMimeoPhotoProject();

} // MacApplication

#endif // PLATFORM(MAC)

#if PLATFORM(IOS_FAMILY)

namespace IOSApplication {

WTF_EXPORT_PRIVATE bool isAmazon();
WTF_EXPORT_PRIVATE bool isAppleApplication();
WTF_EXPORT_PRIVATE bool isCardiogram();
WTF_EXPORT_PRIVATE bool isCrunchyroll();
WTF_EXPORT_PRIVATE bool isDataActivation();
WTF_EXPORT_PRIVATE bool isDoubleDown();
WTF_EXPORT_PRIVATE bool isDumpRenderTree();
WTF_EXPORT_PRIVATE bool isESPNFantasySports();
WTF_EXPORT_PRIVATE bool isEssentialSkeleton();
WTF_EXPORT_PRIVATE bool isEventbrite();
WTF_EXPORT_PRIVATE bool isEvernote();
WTF_EXPORT_PRIVATE bool isFIFACompanion();
WTF_EXPORT_PRIVATE bool isFeedly();
WTF_EXPORT_PRIVATE bool isFirefox();
WTF_EXPORT_PRIVATE bool isHimalaya();
WTF_EXPORT_PRIVATE bool isHoYoLAB();
WTF_EXPORT_PRIVATE bool isIMDb();
WTF_EXPORT_PRIVATE bool isGmail();
WTF_EXPORT_PRIVATE bool isJWLibrary();
WTF_EXPORT_PRIVATE bool isLaBanquePostale();
WTF_EXPORT_PRIVATE bool isLutron();
WTF_EXPORT_PRIVATE bool isMailCompositionService();
WTF_EXPORT_PRIVATE bool isMiniBrowser();
WTF_EXPORT_PRIVATE bool isMobileMail();
WTF_EXPORT_PRIVATE bool isMobileSafari();
WTF_EXPORT_PRIVATE bool isNews();
WTF_EXPORT_PRIVATE bool isNike();
WTF_EXPORT_PRIVATE bool isNoggin();
WTF_EXPORT_PRIVATE bool isOKCupid();
WTF_EXPORT_PRIVATE bool isPaperIO();
WTF_EXPORT_PRIVATE bool isPocketCity();
WTF_EXPORT_PRIVATE bool isSafariViewService();
WTF_EXPORT_PRIVATE bool isStocks();
WTF_EXPORT_PRIVATE bool isTheSecretSocietyHiddenMystery();
WTF_EXPORT_PRIVATE bool isWebBookmarksD();
WTF_EXPORT_PRIVATE bool isWebProcess();
WTF_EXPORT_PRIVATE bool isBackboneApp();
WTF_EXPORT_PRIVATE bool isIBooksStorytime();
WTF_EXPORT_PRIVATE bool isMobileStore();
WTF_EXPORT_PRIVATE bool isMoviStarPlus();
WTF_EXPORT_PRIVATE bool isSpringBoard();
WTF_EXPORT_PRIVATE bool isUNIQLOApp();
WTF_EXPORT_PRIVATE bool isWechat();
WTF_EXPORT_PRIVATE bool isDOFUSTouch();
WTF_EXPORT_PRIVATE bool isMyRideK12();

} // IOSApplication

#endif // PLATFORM(IOS_FAMILY)

} // namespace WTF

using WTF::applicationBundleIdentifier;
using WTF::clearApplicationBundleIdentifierTestingOverride;
using WTF::disableAllSDKAlignedBehaviors;
using WTF::enableAllSDKAlignedBehaviors;
using WTF::linkedOnOrAfterSDKWithBehavior;
using WTF::processIsExtension;
using WTF::SDKAlignedBehavior;
using WTF::sdkAlignedBehaviors;
using WTF::SDKAlignedBehaviors;
using WTF::setApplicationBundleIdentifier;
using WTF::setApplicationBundleIdentifierOverride;
using WTF::setProcessIsExtension;
using WTF::setSDKAlignedBehaviors;

#endif // PLATFORM(COCOA)
