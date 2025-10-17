/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 7, 2024.
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
#if PLATFORM(IOS_FAMILY) && ENABLE(DATA_DETECTION)

#import <UIKit/UIKit.h>

#if USE(APPLE_INTERNAL_SDK)

#import <DataDetectorsUI/DDAction.h>
#if HAVE(LINK_PREVIEW) && USE(UICONTEXTMENU)
#import <DataDetectorsUI/DDContextMenuAction.h>
#endif
#import <DataDetectorsUI/DDDetectionController.h>

#else

#import <pal/spi/cocoa/DataDetectorsCoreSPI.h>

@interface DDAction : NSObject
@end

@interface DDAction ()
- (BOOL)hasUserInterface;
- (NSString *)localizedName;
@property (readonly) NSString *actionUTI;
@end

#if HAVE(LINK_PREVIEW) && USE(UICONTEXTMENU)
@class UIContextMenuConfiguration;
@interface DDContextMenuAction
+ (NSDictionary *)updateContext:(NSDictionary *)context withSourceRect:(CGRect)sourceRect;
+ (UIContextMenuConfiguration *)contextMenuConfigurationForURL:(NSURL *)URL identifier:(NSString *)identifier selectedText:(NSString *)selectedText results:(NSArray *) results inView: (UIView *) view context:(NSDictionary *)context menuIdentifier:(NSString *)menuIdentifier;
+ (UIContextMenuConfiguration *)contextMenuConfigurationWithResult:(DDResultRef)result inView:(UIView *)view context:(NSDictionary *)context menuIdentifier:(NSString *)menuIdentifier;
+ (UIContextMenuConfiguration *)contextMenuConfigurationWithURL:(NSURL *)URL inView:(UIView *)view context:(NSDictionary *)context menuIdentifier:(NSString *)menuIdentifier;
@end
#endif

@protocol DDDetectionControllerInteractionDelegate <NSObject>
@end

@interface DDDetectionController : NSObject <UIActionSheetDelegate>
@end

@interface DDDetectionController ()
+ (DDDetectionController *)sharedController;
+ (NSArray *)tapAndHoldSchemes;
- (void)performAction:(DDAction *)action fromAlertController:(UIAlertController *)alertController interactionDelegate:(id <DDDetectionControllerInteractionDelegate>)interactionDelegate;
- (NSArray *)actionsForURL:(NSURL *)url identifier:(NSString *)identifier selectedText:(NSString *)selectedText results:(NSArray *)results context:(NSDictionary *)context;
- (DDResultRef)resultForURL:(NSURL *)url identifier:(NSString *)identifier selectedText:(NSString *)selectedText results:(NSArray *)results context:(NSDictionary *)context extendedContext:(NSDictionary **)extendedContext;
@end

#endif

@interface DDDetectionController (Staging_48014858)
- (NSString *)titleForURL:(NSURL *)url results:(NSArray *)results context:(NSDictionary *)context;
- (DDAction *)defaultActionForURL:(NSURL *)url results:(NSArray *)results context:(NSDictionary *)context;
- (void)interactionDidStartForURL:(NSURL *)url;
- (BOOL)shouldImmediatelyLaunchDefaultActionForURL:(NSURL *)url;
@end

#endif
