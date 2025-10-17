/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 20, 2023.
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
#import <Cocoa/Cocoa.h>

NS_ASSUME_NONNULL_BEGIN

extern NSString *const MBCRecordingErrorDomain;

typedef NS_ENUM(NSInteger, MBCRecordingErrorCode) {
    MBCRecordingErrorCodeAlreadyRecording = -20031,
    MBCRecordingErrorCodeInvalidWindow    = -20032,
    MBCRecordingErrorCodeInvalidStream    = -20033
};

typedef void (^MBCRecordingBlock) (NSError * _Nullable error);

@class MBCRecordingController;

@protocol MBCRecordingControllerDelegate <NSObject>

/*!
 @abstract recordingController:didStartRecordingWindow:
 @param recordingController Instance of the MBCRecordingController
 @param window The window that is being recorded
 @discussion Called when screen recording did start.
 */
- (void)recordingController:(MBCRecordingController *)recordingController
    didStartRecordingWindow:(NSWindow *)window;

/*!
 @abstract recordingController:didStopRecordingWindow:error:
 @param recordingController Instance of the MBCRecordingController
 @param window The window that was being recorded
 @param error Optional error in the event for when recording failed.
 @discussion Called when screen recording is stopped by either an error
 or user stopping from Control Center module to notify delegate.
 */
- (void)recordingController:(MBCRecordingController *)recordingController
     didStopRecordingWindow:(NSWindow *)window
                      error:(nullable NSError *)error;

@end

@interface MBCRecordingController : NSObject

@property (nonatomic, weak) NSObject<MBCRecordingControllerDelegate> *delegate;

/*!
 @abstract Will be YES if there are any active recordings, NO otherwise.
 */
@property (nonatomic, readonly, getter=isRecording) BOOL recording;

/*!
 @abstract isRecordingWindow:
 @param window The window to check if currently being recorded
 @discussion Returns YES if the window is currently being recorded, NO if not.
 */
- (BOOL)isRecordingWindow:(NSWindow *)window;

/*!
 @abstract startRecordingWindow:completionHandler:
 @param window The window to record
 @param completionHandler Completion handler called after SCStream has started
 @discussion Will start screen recording for the window. Will not start if
 a recording already exists for the same window.
 */
- (void)startRecordingWindow:(NSWindow *)window
           completionHandler:(MBCRecordingBlock)completionHandler;

/*!
 @abstract stopRecordingWindow:completionHandler:
 @param window The window to stop recording
 @param completionHandler Completion handler called after SCStream has stopped
 @discussion Will stop screen recording for the window.
 */
- (void)stopRecordingWindow:(NSWindow *)window
          completionHandler:(MBCRecordingBlock)completionHandler;


/*!
 @abstract cleanupRecordingSessionForWindow:
 @param window The window to clean up after
 @discussion Cleans up the saved data used for recording window.
 */
- (void)cleanupRecordingSessionForWindow:(NSWindow *)window;

@end

NS_ASSUME_NONNULL_END
