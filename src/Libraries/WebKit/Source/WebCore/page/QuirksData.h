/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 23, 2023.
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

namespace WebCore {

struct WEBCORE_EXPORT QuirksData {
    bool isAmazon { false };
    bool isBankOfAmerica { false };
    bool isBing { false };
    bool isCBSSports { false };
    bool isESPN { false };
    bool isFacebook { false };
    bool isGoogleDocs { false };
    bool isGoogleProperty { false };
    bool isGoogleMaps { false };
    bool isNetflix { false };
    bool isSoundCloud { false };
    bool isThesaurus { false };
    bool isVimeo { false };
    bool isWebEx { false };
    bool isYouTube { false };
    bool isZoom { false };

    bool hasBrokenEncryptedMediaAPISupportQuirk { false };
    bool implicitMuteWhenVolumeSetToZero { false };
    bool maybeBypassBackForwardCache { false };
    bool needsBingGestureEventQuirk { false };
    bool needsBodyScrollbarWidthNoneDisabledQuirk { false };
    bool needsCanPlayAfterSeekedQuirk { false };
    bool needsChromeMediaControlsPseudoElementQuirk { false };
    bool needsMozillaFileTypeForDataTransferQuirk { false };
    bool needsResettingTransitionCancelsRunningTransitionQuirk { false };
    bool needsScrollbarWidthThinDisabledQuirk { false };
    bool needsSeekingSupportDisabledQuirk { false };
    bool needsVP9FullRangeFlagQuirk { false };
    bool needsVideoShouldMaintainAspectRatioQuirk { false };
    bool returnNullPictureInPictureElementDuringFullscreenChangeQuirk { false };
    bool shouldAutoplayWebAudioForArbitraryUserGestureQuirk { false };
    bool shouldAvoidResizingWhenInputViewBoundsChangeQuirk { false };
    bool shouldAvoidScrollingWhenFocusedContentIsVisibleQuirk { false };
    bool shouldBypassAsyncScriptDeferring { false };
    bool shouldDisableDataURLPaddingValidation { false };
    bool shouldDisableElementFullscreen { false };
    bool shouldDisableFetchMetadata { false };
    bool shouldDisableLazyIframeLoadingQuirk { false };
    bool shouldDisablePushStateFilePathRestrictions { false };
    bool shouldDisableWritingSuggestionsByDefaultQuirk { false };
    bool shouldDispatchSyntheticMouseEventsWhenModifyingSelectionQuirk { false };
    bool shouldDispatchedSimulatedMouseEventsAssumeDefaultPreventedQuirk { false };
    bool shouldEnableFontLoadingAPIQuirk { false };
    bool shouldExposeShowModalDialog { false };
    bool shouldIgnorePlaysInlineRequirementQuirk { false };
    bool shouldLayOutAtMinimumWindowWidthWhenIgnoringScalingConstraintsQuirk { false };
    bool shouldPreventOrientationMediaQueryFromEvaluatingToLandscapeQuirk { false };
    bool shouldUseLegacySelectPopoverDismissalBehaviorInDataActivationQuirk { false };

    // Requires check at moment of use
    std::optional<bool> needsDisableDOMPasteAccessQuirk;

    std::optional<bool> needsReuseLiveRangeForSelectionUpdateQuirk;

#if PLATFORM(IOS_FAMILY)
    bool mayNeedToIgnoreContentObservation { false };
    bool needsDeferKeyDownAndKeyPressTimersUntilNextEditingCommandQuirk { false };
    bool needsFullscreenDisplayNoneQuirk { false };
    bool needsFullscreenObjectFitQuirk { false };
    bool needsGMailOverflowScrollQuirk { false };
    bool needsGoogleMapsScrollingQuirk { false };
    bool needsIPadSkypeOverflowScrollQuirk { false };
    bool needsPreloadAutoQuirk { false };
    bool needsScriptToEvaluateBeforeRunningScriptFromURLQuirk { false };
    bool needsYouTubeMouseOutQuirk { false };
    bool needsYouTubeOverflowScrollQuirk { false };
    bool shouldAvoidPastingImagesAsWebContent { false };
    bool shouldDisablePointerEventsQuirk { false };
    bool shouldEnableApplicationCacheQuirk { false };
    bool shouldIgnoreAriaForFastPathContentObservationCheckQuirk { false };
    bool shouldNavigatorPluginsBeEmpty { false };
    bool shouldSilenceWindowResizeEventsDuringApplicationSnapshotting { false };
    bool shouldSuppressAutocorrectionAndAutocapitalizationInHiddenEditableAreasQuirk { false };
    bool shouldSynthesizeTouchEventsAfterNonSyntheticClickQuirk { false };
    bool shouldTreatAddingMouseOutEventListenerAsContentChange { false };
#endif // PLATFORM(IOS_FAMILY)

#if PLATFORM(IOS)
    bool needsGetElementsByNameQuirk { false };
#endif

#if PLATFORM(IOS) || PLATFORM(VISION)
    bool allowLayeredFullscreenVideos { false };
    bool shouldSilenceMediaQueryListChangeEvents { false };
    bool shouldSilenceResizeObservers { false };
#endif

#if PLATFORM(VISION)
    bool shouldDisableFullscreenVideoAspectRatioAdaptiveSizingQuirk { false };
#endif

#if PLATFORM(MAC)
    bool isNeverRichlyEditableForTouchBarQuirk { false };
    bool isTouchBarUpdateSuppressedForHiddenContentEditableQuirk { false };
    bool needsFormControlToBeMouseFocusableQuirk { false };
    bool needsPrimeVideoUserSelectNoneQuirk { false };
    bool needsZomatoEmailLoginLabelQuirk { false };
    bool shouldAvoidStartingSelectionOnMouseDown { false };
#endif

#if ENABLE(DESKTOP_CONTENT_MODE_QUIRKS)
    bool needsZeroMaxTouchPointsQuirk { false };
    bool shouldHideCoarsePointerCharacteristicsQuirk { false };
#endif

#if ENABLE(FLIP_SCREEN_DIMENSIONS_QUIRKS)
    bool shouldFlipScreenDimensionsQuirk { false };
#endif

#if ENABLE(MEDIA_STREAM)
    bool shouldDisableImageCaptureQuirk { false };
    bool shouldEnableLegacyGetUserMediaQuirk { false };
    bool shouldEnableSpeakerSelectionPermissionsPolicyQuirk { false };
#endif

#if ENABLE(META_VIEWPORT)
    bool shouldIgnoreViewportArgumentsToAvoidExcessiveZoomQuirk { false };
#endif

#if ENABLE(TEXT_AUTOSIZING)
    bool shouldIgnoreTextAutoSizingQuirk { false };
#endif

#if ENABLE(TOUCH_EVENTS)
    enum class ShouldDispatchSimulatedMouseEvents : uint8_t {
        Unknown,
        No,
        DependingOnTargetFor_mybinder_org,
        Yes,
    };
    ShouldDispatchSimulatedMouseEvents shouldDispatchSimulatedMouseEventsQuirk { ShouldDispatchSimulatedMouseEvents::Unknown };
    bool shouldDispatchPointerOutAfterHandlingSyntheticClick { false };
    bool shouldPreventDispatchOfTouchEventQuirk { false };
#endif

#if ENABLE(FULLSCREEN_API) && ENABLE(VIDEO_PRESENTATION_MODE)
    bool blocksEnteringStandardFullscreenFromPictureInPictureQuirk { false };
    bool blocksReturnToFullscreenFromPictureInPictureQuirk { false };
#endif

#if ENABLE(VIDEO_PRESENTATION_MODE)
    bool requiresUserGestureToLoadInPictureInPictureQuirk { false };
    bool requiresUserGestureToPauseInPictureInPictureQuirk { false };
    bool shouldDelayFullscreenEventWhenExitingPictureInPictureQuirk { false };
    bool shouldDisableEndFullscreenEventWhenEnteringPictureInPictureFromFullscreenQuirk { false };
#endif

    bool needsNowPlayingFullscreenSwapQuirk { false };
    bool needsWebKitMediaTextTrackDisplayQuirk { false };
};

} // namespace WebCore
