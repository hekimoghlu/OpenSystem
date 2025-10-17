/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 6, 2022.
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
#import <WebKit/WKFoundation.h>

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

WK_API_AVAILABLE(macos(10.13), ios(11.0))
@protocol WKURLSchemeTask <NSObject>

/*! @abstract The request to load for this task.
 */
@property (nonatomic, readonly, copy) NSURLRequest *request;

/*! @abstract Set the current response object for the task.
 @param response The response to use.
 @discussion This method must be called at least once for each URL scheme handler task.
 Cross-origin requests require CORS header fields.
 An exception will be thrown if you try to send a new response object after the task has already been completed.
 An exception will be thrown if your app has been told to stop loading this task via the registered WKURLSchemeHandler object.
 */
- (void)didReceiveResponse:(NSURLResponse *)response;

/*! @abstract Add received data to the task.
 @param data The data to add.
 @discussion After a URL scheme handler task's final response object is received you should
 start sending it data.
 Each time this method is called the data you send will be appended to all previous data.
 An exception will be thrown if you try to send the task any data before sending it a response.
 An exception will be thrown if you try to send the task any data after the task has already been completed.
 An exception will be thrown if your app has been told to stop loading this task via the registered WKURLSchemeHandler object.
 */
- (void)didReceiveData:(NSData *)data;

/*! @abstract Mark the task as successfully completed.
 @discussion An exception will be thrown if you try to finish the task before sending it a response.
 An exception will be thrown if you try to mark a task completed after it has already been marked completed or failed.
 An exception will be thrown if your app has been told to stop loading this task via the registered WKURLSchemeHandler object.
 */
- (void)didFinish;

/*! @abstract Mark the task as failed.
 @param error A description of the error that caused the task to fail.
 @discussion  An exception will be thrown if you try to mark a task failed after it has already been marked completed or failed.
 An exception will be thrown if your app has been told to stop loading this task via the registered WKURLSchemeHandler object.
 */
- (void)didFailWithError:(NSError *)error;

@end

NS_ASSUME_NONNULL_END
