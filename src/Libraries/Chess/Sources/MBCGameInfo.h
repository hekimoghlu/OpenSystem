/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 7, 2023.
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

#import "MBCBoard.h"

@class MBCDocument;

@interface MBCGameInfo : NSObject
{
    MBCDocument *  fDocument;
    IBOutlet NSPanel *      fEditSheet;
    IBOutlet NSPanel *      fPrefsSheet;
    IBOutlet NSTableView *  fMoveList;
    IBOutlet MBCBoard *     fBoard;
    IBOutlet NSPopUpButton *fPrimaryVoiceMenu;
    IBOutlet NSPopUpButton *fAlternateVoiceMenu;
    IBOutlet NSPopUpButton *fBoardStyle;
    IBOutlet NSPopUpButton *fPieceStyle;
    IBOutlet NSTextField   *fBoardStyleText;
    IBOutlet NSTextField   *fPieceStyleText;
    IBOutlet NSLayoutConstraint *fBoardStyleTrailingConstraint;
    IBOutlet NSLayoutConstraint *fPieceStyleTrailingConstraint;

    NSMutableDictionary *   fStyleLocMap;
    NSMutableDictionary *   fEditedProperties;
    
    int                     fRows;
    BOOL                    fHasObservers;
}

@property (nonatomic,assign)    MBCDocument *   document;
@property (nonatomic)           BOOL            whiteEditable;
@property (nonatomic)           BOOL            blackEditable;
@property (nonatomic,readonly)  NSTableView *   moveList;

- (void) startGame:(MBCVariant)variant playing:(MBCSide)sideToPlay;
- (int)numberOfRowsInTableView:(NSTableView *)aTableView;
- (id)tableView:(NSTableView *)aTableView objectValueForTableColumn:(NSTableColumn *)aTableColumn row:(int)rowIndex;
- (IBAction) editInfo:(id)sender;
- (void)editPreferencesForWindow:(NSWindow *)window hidePiecesStyle:(BOOL)hidePiecesStyle;
- (IBAction) cancelProperties:(id)sender;
- (IBAction) updateProperties:(id)sender;
- (IBAction) updateVoices:(id)sender;
- (IBAction) updateStyles:(id)sender;
- (NSString *)gameTitle;
- (NSString *)describeMove:(int)move;
- (void) removeChessObservers;

@end

// Local Variables:
// mode:ObjC
// End:
