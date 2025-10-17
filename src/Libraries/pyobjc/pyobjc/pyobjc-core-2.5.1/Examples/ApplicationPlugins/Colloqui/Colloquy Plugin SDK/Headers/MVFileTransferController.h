/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 3, 2022.
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

#import <Foundation/NSObject.h>
#import <Foundation/NSDate.h>
#import <AppKit/NSNibDeclarations.h>

NSString *MVPrettyFileSize( unsigned long long size );
NSString *MVReadableTime( NSTimeInterval date, BOOL longFormat );

@class NSPanel;
@class NSProgressIndicator;
@class NSTextField;
@class NSTableView;
@class NSMutableArray;
@class NSRecursiveLock;
@class NSTimer;

@interface MVFileTransferController : NSWindowController {
@private
	IBOutlet NSProgressIndicator *progressBar;
	IBOutlet NSTextField *transferStatus;
	IBOutlet NSTableView *currentFiles;
	NSMutableArray *_transferStorage;
	NSMutableArray *_calculationItems;
	NSTimer *_updateTimer;
	NSSet *_safeFileExtentions;
}
+ (NSString *) userPreferredDownloadFolder;
+ (void) setUserPreferredDownloadFolder:(NSString *) path;

+ (MVFileTransferController *) defaultManager;

- (IBAction) showTransferManager:(id) sender;
- (IBAction) hideTransferManager:(id) sender;

- (void) downloadFileAtURL:(NSURL *) url toLocalFile:(NSString *) path;
- (void) addFileTransfer:(id) trtansfer;

- (IBAction) stopSelectedTransfer:(id) sender;
- (IBAction) clearFinishedTransfers:(id) sender;
- (IBAction) revealSelectedFile:(id) sender;
@end
