/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 24, 2023.
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
#import <WebKit/WKWebExtensionCommand.h>

@interface WKWebExtensionCommand ()

/*!
 @abstract Represents the shortcut for the web extension, formatted according to web extension specification.
 @discussion Provides a string representation of the shortcut, incorporating any customizations made to the ``activationKey``
 and ``modifierFlags`` properties. It will be empty if no shortcut is defined for the command.
 */
@property (nonatomic, readonly, copy) NSString *_shortcut;

/*!
 @abstract Represents the user visible shortcut for the web extension, formatted according to the system.
 @discussion Provides a string representation of the shortcut, incorporating any customizations made to the ``activationKey``
 and ``modifierFlags`` properties. It will be empty if no shortcut is defined for the command.
 */
@property (nonatomic, readonly, copy) NSString *_userVisibleShortcut;

/*! @abstract Represents whether or not this command is the action command for the extension. */
@property (nonatomic, readonly) BOOL _isActionCommand;

#if TARGET_OS_OSX
/*!
 @abstract Determines whether an event matches the command's activation key and modifier flags.
 @discussion This method can be used to check if a given keyboard event corresponds to the command's activation key and modifiers, if any.
 The app can use this during event handling in the app, without showing the command in a menu.
 @param event The event to be checked against the command's activation key and modifiers.
 @result A Boolean value indicating whether the event matches the command's shortcut.
 */
- (BOOL)_matchesEvent:(NSEvent *)event;
#endif // TARGET_OS_OSX

@end
