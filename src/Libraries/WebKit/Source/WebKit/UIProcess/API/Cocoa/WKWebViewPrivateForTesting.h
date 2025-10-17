/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 29, 2025.
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
#import <Foundation/Foundation.h>

#if TARGET_OS_IPHONE
#import <WebKit/WKWebViewPrivateForTestingIOS.h>
#else
#import <WebKit/WKWebViewPrivateForTestingMac.h>
#endif

NS_ASSUME_NONNULL_BEGIN

typedef enum {
    WKWebViewAudioRoutingArbitrationStatusNone,
    WKWebViewAudioRoutingArbitrationStatusPending,
    WKWebViewAudioRoutingArbitrationStatusActive,
} WKWebViewAudioRoutingArbitrationStatus;

struct WKAppPrivacyReportTestingData {
    BOOL hasLoadedAppInitiatedRequestTesting;
    BOOL hasLoadedNonAppInitiatedRequestTesting;
    BOOL didPerformSoftUpdate;
};

@class _WKNowPlayingMetadata;
@protocol _WKMediaSessionCoordinator;

@interface WKWebView (WKTesting)

@property (nonatomic, readonly) NSString *_caLayerTreeAsText;

- (NSString*)_scrollbarStateForScrollingNodeID:(uint64_t)scrollingNodeID processID:(uint64_t)processID isVertical:(bool)isVertical;

- (void)_addEventAttributionWithSourceID:(uint8_t)sourceID destinationURL:(NSURL *)destination sourceDescription:(NSString *)sourceDescription purchaser:(NSString *)purchaser reportEndpoint:(NSURL *)reportEndpoint optionalNonce:(nullable NSString *)nonce applicationBundleID:(NSString *)bundleID ephemeral:(BOOL)ephemeral WK_API_AVAILABLE(macos(13.0), ios(16.0));

- (void)_setPageScale:(CGFloat)scale withOrigin:(CGPoint)origin;
- (CGFloat)_pageScale;

- (void)_setContinuousSpellCheckingEnabledForTesting:(BOOL)enabled;
- (void)_setGrammarCheckingEnabledForTesting:(BOOL)enabled;
- (NSDictionary *)_contentsOfUserInterfaceItem:(NSString *)userInterfaceItem;

- (void)_requestActiveNowPlayingSessionInfo:(void(^)(BOOL, BOOL, NSString*, double, double, NSInteger))callback;
- (void)_setNowPlayingMetadataObserver:(void(^)(_WKNowPlayingMetadata *))observer;

- (void)_doAfterNextPresentationUpdateWithoutWaitingForAnimatedResizeForTesting:(void (^)(void))updateBlock;

- (void)_disableBackForwardSnapshotVolatilityForTesting;

- (void)_denyNextUserMediaRequest;
@property (nonatomic, setter=_setMediaCaptureReportingDelayForTesting:) double _mediaCaptureReportingDelayForTesting WK_API_AVAILABLE(macos(12.0), ios(15.0));
@property (nonatomic, readonly) BOOL _wirelessVideoPlaybackDisabled;

- (void)_setIndexOfGetDisplayMediaDeviceSelectedForTesting:(nullable NSNumber *)index;
- (void)_setSystemCanPromptForGetDisplayMediaForTesting:(BOOL)canPrompt;

- (BOOL)_beginBackSwipeForTesting;
- (BOOL)_completeBackSwipeForTesting;
- (void)_resetNavigationGestureStateForTesting;

- (void)_setShareSheetCompletesImmediatelyWithResolutionForTesting:(BOOL)resolved;

- (void)_didShowContextMenu;
- (void)_didDismissContextMenu;

- (void)_resetInteraction;

- (BOOL)_shouldBypassGeolocationPromptForTesting;

- (void)_didPresentContactPicker;
- (void)_didDismissContactPicker;
- (void)_dismissContactPickerWithContacts:(NSArray *)contacts;

@property (nonatomic, setter=_setScrollingUpdatesDisabledForTesting:) BOOL _scrollingUpdatesDisabledForTesting;
@property (nonatomic, readonly) NSString *_scrollingTreeAsText;

@property (nonatomic, readonly) pid_t _networkProcessIdentifier;

@property (nonatomic, readonly) unsigned long _countOfUpdatesWithLayerChanges;

- (void)_processWillSuspendForTesting:(void (^)(void))completionHandler;
- (void)_processWillSuspendImminentlyForTesting;
- (void)_processDidResumeForTesting;
@property (nonatomic, readonly) BOOL _hasServiceWorkerBackgroundActivityForTesting;
@property (nonatomic, readonly) BOOL _hasServiceWorkerForegroundActivityForTesting;
- (void)_setThrottleStateForTesting:(int)type;

- (void)_doAfterProcessingAllPendingMouseEvents:(dispatch_block_t)action;

+ (void)_setApplicationBundleIdentifier:(NSString *)bundleIdentifier;
+ (void)_clearApplicationBundleIdentifierTestingOverride;

- (BOOL)_hasSleepDisabler;
- (WKWebViewAudioRoutingArbitrationStatus)_audioRoutingArbitrationStatus;
- (double)_audioRoutingArbitrationUpdateTime;

- (void)_doAfterActivityStateUpdate:(void (^)(void))completionHandler;

- (NSNumber *)_suspendMediaPlaybackCounter WK_API_AVAILABLE(macos(12.0), ios(15.0));

- (void)_setPrivateClickMeasurementOverrideTimerForTesting:(BOOL)overrideTimer completionHandler:(void(^)(void))completionHandler WK_API_AVAILABLE(macos(12.0), ios(15.0));
- (void)_setPrivateClickMeasurementAttributionReportURLsForTesting:(NSURL *)sourceURL destinationURL:(NSURL *)destinationURL completionHandler:(void(^)(void))completionHandler WK_API_AVAILABLE(macos(12.0), ios(15.0));
- (void)_setPrivateClickMeasurementAttributionTokenPublicKeyURLForTesting:(NSURL *)url completionHandler:(void(^)(void))completionHandler WK_API_AVAILABLE(macos(12.0), ios(15.0));
- (void)_setPrivateClickMeasurementAttributionTokenSignatureURLForTesting:(NSURL *)url completionHandler:(void(^)(void))completionHandler WK_API_AVAILABLE(macos(12.0), ios(15.0));
- (void)_setPrivateClickMeasurementAppBundleIDForTesting:(NSString *)appBundleID completionHandler:(void(^)(void))completionHandler WK_API_AVAILABLE(macos(13.0), ios(16.0));
- (void)_dumpPrivateClickMeasurement:(void(^)(NSString *))completionHandler WK_API_AVAILABLE(macos(13.0), ios(16.0));

- (void)_lastNavigationWasAppInitiated:(void(^)(BOOL))completionHandler;
- (void)_appPrivacyReportTestingData:(void(^)(struct WKAppPrivacyReportTestingData data))completionHandler;
- (void)_clearAppPrivacyReportTestingData:(void(^)(void))completionHandler;

- (void)_createMediaSessionCoordinatorForTesting:(id <_WKMediaSessionCoordinator>)privateCoordinator completionHandler:(void(^)(BOOL))completionHandler;
- (void)_gpuToWebProcessConnectionCountForTesting:(void(^)(NSUInteger))completionHandler WK_API_AVAILABLE(macos(13.0), ios(16.0));

- (void)_isLayerTreeFrozenForTesting:(void (^)(BOOL frozen))completionHandler WK_API_AVAILABLE(macos(13.0), ios(16.0));

- (void)_computePagesForPrinting:(_WKFrameHandle *)handle completionHandler:(void(^)(void))completionHandler WK_API_AVAILABLE(macos(13.0), ios(16.0));

- (void)_setConnectedToHardwareConsoleForTesting:(BOOL)connected;

- (void)_setSystemPreviewCompletionHandlerForLoadTesting:(void(^)(bool))completionHandler;

@property (nonatomic, readonly) BOOL _isLoggerEnabledForTesting;

- (void)_terminateIdleServiceWorkersForTesting;

- (void)_getNotifyStateForTesting:(NSString *)notificationName completionHandler:(void(^)(NSNumber * _Nullable))completionHandler WK_API_AVAILABLE(macos(WK_MAC_TBA), ios(WK_IOS_TBA), visionos(WK_XROS_TBA));

@property (nonatomic, readonly) BOOL _hasAccessibilityActivityForTesting;
@end

typedef NS_ENUM(NSInteger, _WKMediaSessionReadyState) {
    WKMediaSessionReadyStateHaveNothing,
    WKMediaSessionReadyStateHaveMetadata,
    WKMediaSessionReadyStateHaveCurrentData,
    WKMediaSessionReadyStateHaveFutureData,
    WKMediaSessionReadyStateHaveEnoughData
};

typedef NS_ENUM(NSInteger, _WKMediaSessionPlaybackState) {
    WKMediaSessionPlaybackStateNone,
    WKMediaSessionPlaybackStatePaused,
    WKMediaSessionPlaybackStatePlaying
};

typedef NS_ENUM(NSInteger, _WKMediaSessionCoordinatorState) {
    WKMediaSessionCoordinatorStateWaiting,
    WKMediaSessionCoordinatorStateJoined,
    WKMediaSessionCoordinatorStateClosed
};

struct _WKMediaPositionState {
    double duration;
    double playbackRate;
    double position;
};

@protocol _WKMediaSessionCoordinatorDelegate <NSObject>
- (void)seekSessionToTime:(double)time withCompletion:(void(^)(BOOL))completionHandler;
- (void)playSessionWithCompletion:(void(^)(BOOL))completionHandler;
- (void)pauseSessionWithCompletion:(void(^)(BOOL))completionHandler;
- (void)setSessionTrack:(NSString*)trackIdentifier withCompletion:(void(^)(BOOL))completionHandler;
- (void)coordinatorStateChanged:(_WKMediaSessionCoordinatorState)state;
@end

@protocol _WKMediaSessionCoordinator <NSObject>
@property (nullable, weak) id <_WKMediaSessionCoordinatorDelegate> delegate;
@property (nonatomic, readonly) NSString * _Nonnull identifier;
- (void)joinWithCompletion:(void(^ _Nonnull)(BOOL))completionHandler;
- (void)leave;
- (void)seekTo:(double)time withCompletion:(void(^ _Nonnull)(BOOL))completionHandler;
- (void)playWithCompletion:(void(^ _Nonnull)(BOOL))completionHandler;
- (void)pauseWithCompletion:(void(^ _Nonnull)(BOOL))completionHandler;
- (void)setTrack:(NSString *_Nonnull)trackIdentifier withCompletion:(void(^ _Nonnull)(BOOL))completionHandler;
- (void)positionStateChanged:(struct _WKMediaPositionState * _Nullable)state;
- (void)readyStateChanged:(_WKMediaSessionReadyState)state;
- (void)playbackStateChanged:(_WKMediaSessionPlaybackState)state;
- (void)trackIdentifierChanged:(NSString *)trackIdentifier;
@end

WK_EXTERN
@interface _WKNowPlayingMetadata : NSObject
@property (nonatomic, copy) NSString *title;
@property (nonatomic, copy) NSString *artist;
@property (nonatomic, copy) NSString *album;
@property (nonatomic, copy) NSString *sourceApplicationIdentifier;
@end

NS_ASSUME_NONNULL_END
