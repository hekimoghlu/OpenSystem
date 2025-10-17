/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 6, 2024.
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
#import <WebKit/WKFoundation.h>

WK_HEADER_AUDIT_BEGIN(nullability, sendability)

/*! @abstract Indicates a ``WKWebExtensionMessagePort`` error. */
WK_API_AVAILABLE(macos(WK_MAC_TBA), ios(WK_IOS_TBA), visionos(WK_XROS_TBA))
WK_EXTERN NSErrorDomain const WKWebExtensionMessagePortErrorDomain NS_SWIFT_NAME(WKWebExtensionMessagePort.errorDomain) NS_SWIFT_NONISOLATED;

/*!
 @abstract Constants used by ``NSError`` to indicate errors in the ``WKWebExtensionMessagePort`` domain.
 @constant WKWebExtensionMessagePortErrorUnknown  Indicates that an unknown error occurred.
 @constant WKWebExtensionMessagePortErrorNotConnected  Indicates that the message port is disconnected.
 @constant WKWebExtensionMessagePortErrorMessageInvalid Indicates that the message is invalid. The message must be an object that is JSON-serializable.
 */
typedef NS_ERROR_ENUM(WKWebExtensionMessagePortErrorDomain, WKWebExtensionMessagePortError) {
    WKWebExtensionMessagePortErrorUnknown = 1,
    WKWebExtensionMessagePortErrorNotConnected,
    WKWebExtensionMessagePortErrorMessageInvalid,
} NS_SWIFT_NAME(WKWebExtensionMessagePort.Error) WK_API_AVAILABLE(macos(WK_MAC_TBA), ios(WK_IOS_TBA), visionos(WK_XROS_TBA));

/*!
 @abstract A ``WKWebExtensionMessagePort`` object manages message-based communication with a web extension.
 @discussion Contains properties and methods to handle message exchanges with a web extension.
*/
WK_CLASS_AVAILABLE(macos(WK_MAC_TBA), ios(WK_IOS_TBA), visionos(WK_XROS_TBA))
WK_SWIFT_UI_ACTOR NS_SWIFT_NAME(WKWebExtension.MessagePort)
@interface WKWebExtensionMessagePort : NSObject

+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

/*!
 @abstract The unique identifier for the app to which this port should be connected.
 @discussion This identifier is provided by the web extension and may or may not be used by the app.
 It's up to the app to decide how to interpret this identifier.
 */
@property (nonatomic, readonly, nullable) NSString *applicationIdentifier;

/*!
 @abstract The block to be executed when a message is received from the web extension.
 @discussion An optional block to be invoked when a message is received, taking two parameters: the message and an optional error.
 */
@property (nonatomic, copy, nullable) void (^messageHandler)(id _Nullable message, NSError * _Nullable error);

/*!
 @abstract The block to be executed when the port disconnects.
 @discussion An optional block to be invoked when the port disconnects, taking an optional error that indicates if the disconnection was caused by an error.
 */
@property (nonatomic, copy, nullable) void (^disconnectHandler)(NSError * _Nullable error);

/*! @abstract Indicates whether the message port is disconnected. */
@property (nonatomic, readonly, getter=isDisconnected) BOOL disconnected;

/*!
 @abstract Sends a message to the connected web extension.
 @param message The JSON-serializable message to be sent.
 @param completionHandler An optional block to be invoked after the message is sent, taking an optional error.
 @note The message must be JSON-serializable according to ``NSJSONSerialization``.
 */
- (void)sendMessage:(nullable id)message completionHandler:(void (^ _Nullable)(NSError * _Nullable error))completionHandler NS_SWIFT_NAME(sendMessage(_:completionHandler:));

/*! @abstract Disconnects the port, terminating all further messages. */
- (void)disconnect;

/*!
 @abstract Disconnects the port, terminating all further messages with an optional error.
 @param error An optional error indicating the reason for disconnection.
 */
- (void)disconnectWithError:(nullable NSError *)error NS_SWIFT_NAME(disconnect(throwing:));

@end

WK_HEADER_AUDIT_END(nullability, sendability)
