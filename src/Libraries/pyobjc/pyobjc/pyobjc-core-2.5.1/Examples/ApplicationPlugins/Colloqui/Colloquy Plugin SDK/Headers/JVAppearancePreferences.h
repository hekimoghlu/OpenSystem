/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 25, 2025.
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

#import <AppKit/NSNibDeclarations.h>
#import "NSPreferences.h"

@class WebView;
@class NSPopUpButton;
@class NSTextView;
@class JVFontPreviewField;
@class NSTextField;
@class NSStepper;
@class NSSet;
@class NSDrawer;
@class NSTableView;
@class JVStyle;

@interface JVAppearancePreferences : NSPreferencesModule {
	IBOutlet WebView *preview;
	IBOutlet NSPopUpButton *styles;
	IBOutlet NSPopUpButton *emoticons;
	IBOutlet JVFontPreviewField *standardFont;
	IBOutlet NSTextField *minimumFontSize;
	IBOutlet NSStepper *minimumFontSizeStepper;
	IBOutlet NSTextField *baseFontSize;
	IBOutlet NSStepper *baseFontSizeStepper;
	IBOutlet NSButton *useStyleFont;
	IBOutlet NSDrawer *optionsDrawer;
	IBOutlet NSTableView *optionsTable;
	IBOutlet NSPanel *newVariantPanel;
	IBOutlet NSTextField *newVariantName;
	BOOL _variantLocked;
	BOOL _alertDisplayed;
	JVStyle *_style;
	NSSet *_emoticonBundles;
	NSMutableArray *_styleOptions;
	NSString *_userStyle;
}
- (void) selectStyleWithIdentifier:(NSString *) identifier;
- (void) selectEmoticonsWithIdentifier:(NSString *) identifier;

- (void) setStyle:(JVStyle *) style;

- (void) changePreferences;

- (IBAction) changeBaseFontSize:(id) sender;
- (IBAction) changeMinimumFontSize:(id) sender;

- (IBAction) changeDefaultChatStyle:(id) sender;

- (IBAction) noGraphicEmoticons:(id) sender;
- (IBAction) changeDefaultEmoticons:(id) sender;

- (IBAction) changeUseStyleFont:(id) sender;

- (IBAction) showOptions:(id) sender;

- (void) updateChatStylesMenu;
- (void) updateEmoticonsMenu;
- (void) updatePreview;
- (void) updateVariant;

- (void) parseStyleOptions;
- (NSString *) valueOfProperty:(NSString *) property forSelector:(NSString *) selector inStyle:(NSString *) style;
- (void) setStyleProperty:(NSString *) property forSelector:(NSString *) selector toValue:(NSString *) value;
- (void) setUserStyle:(NSString *) style;
- (void) saveStyleOptions;

- (void) showNewVariantSheet;
- (IBAction) closeNewVariantSheet:(id) sender;
- (IBAction) createNewVariant:(id) sender;
@end
