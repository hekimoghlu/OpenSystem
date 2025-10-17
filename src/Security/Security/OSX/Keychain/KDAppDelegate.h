/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 19, 2022.
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
#import "KDSecItems.h"
#import "KDSecCircle.h"

@interface KDAppDelegate : NSObject <NSApplicationDelegate>

@property (assign) IBOutlet NSWindow *window;
@property (assign) IBOutlet NSTableView *itemTable;
@property (assign) IBOutlet NSTextFieldCell *itemTableTitle;
@property (retain) id<NSTableViewDataSource> itemDataSource;

@property (assign) IBOutlet NSButton *enableKeychainSyncing;
@property (assign) IBOutlet NSTextFieldCell *circleStatusCell;
@property (assign) IBOutlet NSTextFieldCell *peerCountCell;
@property (assign) IBOutlet NSTextView *peerTextList;
@property (assign) IBOutlet NSTextFieldCell *applicantCountCell;
@property (assign) IBOutlet NSTextView *applicantTextList;
@property (assign) IBOutlet NSProgressIndicator *syncSpinner;

@property (retain) KDSecCircle *circle;

@property (retain) NSMutableArray *stuffNotToLeak;

-(IBAction)enableKeychainSyncingClicked:(id)sender;
@end
