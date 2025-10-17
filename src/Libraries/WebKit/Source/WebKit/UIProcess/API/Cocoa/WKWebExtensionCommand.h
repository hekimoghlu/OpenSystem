/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 13, 2024.
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

#if TARGET_OS_IPHONE
#import <UIKit/UIKeyCommand.h>
#endif

@class WKWebExtensionContext;

WK_HEADER_AUDIT_BEGIN(nullability, sendability)

/*!
 @abstract A ``WKWebExtensionCommand`` object encapsulates the properties for an individual web extension command.
 @discussion Provides access to command properties such as a unique identifier, a descriptive title, and shortcut keys. Commands
 can be used by a web extension to perform specific actions within a web extension context, such toggling features, or interacting with
 web content. These commands enhance the functionality of the extension by allowing users to invoke actions quickly.
 */
WK_CLASS_AVAILABLE(macos(WK_MAC_TBA), ios(WK_IOS_TBA), visionos(WK_XROS_TBA))
WK_SWIFT_UI_ACTOR NS_SWIFT_NAME(WKWebExtension.Command)
@interface WKWebExtensionCommand : NSObject

+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

/*! @abstract The web extension context associated with the command. */
@property (nonatomic, readonly, weak) WKWebExtensionContext *webExtensionContext;

/*! @abstract A unique identifier for the command. */
@property (nonatomic, readonly, copy) NSString *identifier NS_SWIFT_NAME(id);

/*!
 @abstract Descriptive title for the command aiding discoverability.
 @discussion This title can be displayed in user interface elements such as keyboard shortcuts lists or menu items to help users understand its purpose.
 */
@property (nonatomic, readonly, copy) NSString *title;

/*!
 @abstract The primary key used to trigger the command, distinct from any modifier flags.
 @discussion This property can be customized within the app to avoid conflicts with existing shortcuts or to enable user personalization.
 It should accurately represent the activation key as used by the app, which the extension can use to display the complete shortcut in its interface.
 If no shortcut is desired for the command, the property should be set to `nil`. This value should be saved and restored as needed by the app.
 */
@property (nonatomic, nullable, copy) NSString *activationKey;

/*!
 @abstract The modifier flags used with the activation key to trigger the command.
 @discussion This property can be customized within the app to avoid conflicts with existing shortcuts or to enable user personalization. It
 should accurately represent the modifier keys as used by the app, which the extension can use to display the complete shortcut in its interface.
 If no modifiers are desired for the command, the property should be set to `0`. This value should be saved and restored as needed by the app.
 */
#if TARGET_OS_IPHONE
@property (nonatomic) UIKeyModifierFlags modifierFlags;
#else
@property (nonatomic) NSEventModifierFlags modifierFlags;
#endif

/*!
 @abstract A menu item representation of the web extension command for use in menus.
 @discussion Provides a representation of the web extension command as a menu item to display in the app.
 Selecting the menu item will perform the command, offering a convenient and visual way for users to execute this web extension command.
 */
#if TARGET_OS_IPHONE
@property (nonatomic, readonly, copy) UIMenuElement *menuItem;
#else
@property (nonatomic, readonly, copy) NSMenuItem *menuItem;
#endif

#if TARGET_OS_IPHONE
/*!
 @abstract A key command representation of the web extension command for use in the responder chain.
 @discussion Provides a ``UIKeyCommand`` instance representing the web extension command, ready for integration in the app.
 The property is `nil` if no shortcut is defined. Otherwise, the key command is fully configured with the necessary input key and modifier flags
 to perform the associated command upon activation. It can be included in a view controller or other responder's ``keyCommands`` property, enabling
 keyboard activation and discoverability of the web extension command.
 */
@property (nonatomic, readonly, copy, nullable) UIKeyCommand *keyCommand;
#endif // TARGET_OS_IPHONE

@end

WK_HEADER_AUDIT_END(nullability, sendability)
