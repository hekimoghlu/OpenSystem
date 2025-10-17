/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 24, 2023.
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
#import "MBCGameInfo.h"
#import "MBCBoardWin.h"
#import "MBCController.h"
#import "MBCPlayer.h"
#import "MBCDocument.h"
#import "MBCUserDefaults.h"

#import <GameKit/GameKit.h>


#include <sys/types.h>
#include <regex.h>
#include <algorithm>

NSArray *   sVoices;

@implementation MBCGameInfo

@synthesize document = fDocument;
@synthesize moveList = fMoveList;
@synthesize whiteEditable, blackEditable;

NSString * kMBCShowMoveInTitle  = @"MBCShowMoveInTitle";
//
// Obsolete: Parsing the human name serves no purpose and carries some risk
//
NSString * kMBCHumanFirst		= @"MBCHumanFirst";
NSString * kMBCHumanLast		= @"MBCHumanLast";

+ (void)initialize
{
	NSUserDefaults * 	userDefaults = [NSUserDefaults standardUserDefaults];
	NSString * humanName;

	//
	// Deal with legacy user name default representation
	//
	if ([userDefaults stringForKey:kMBCHumanFirst]) {
		humanName = [NSString stringWithFormat:@"%@ %@", 
							  [userDefaults stringForKey:kMBCHumanFirst],
							  [userDefaults stringForKey:kMBCHumanLast]];
		[userDefaults removeObjectForKey:kMBCHumanFirst];
		[userDefaults removeObjectForKey:kMBCHumanLast];
	} else
		humanName = NSFullUserName();

	NSString *		city 	= @"?";
	NSString *		country	= @"?";

	NSString * event = 
		[NSLocalizedString(@"casual_game", @"Casual Game") retain];

	NSDictionary * defaults = 
		[NSDictionary 
			dictionaryWithObjectsAndKeys:
				 humanName, kMBCHumanName,
			          city, kMBCGameCity,
   			       country, kMBCGameCountry,
			         event, kMBCGameEvent,
			nil];
	[userDefaults registerDefaults: defaults];
    sVoices = [[NSSpeechSynthesizer availableVoices] retain];
}

const int kNumFixedMenuItems = 2;

- (void)loadVoiceMenu:(id)menu
{
    for (NSString * voiceIdentifier in sVoices) 
        [menu addItemWithTitle:[[NSSpeechSynthesizer attributesForVoice:voiceIdentifier] objectForKey:NSVoiceName]];
}

- (NSString *)voiceAtIndex:(NSUInteger)menuIndex
{
    if (menuIndex < kNumFixedMenuItems)
        return nil;
    else 
        return [sVoices objectAtIndex:menuIndex-kNumFixedMenuItems];
}

- (NSUInteger)indexForVoice:(NSString *)voiceId
{
    return [voiceId length] ? [sVoices indexOfObject:voiceId]+kNumFixedMenuItems : 0;
}

- (NSString *) localizedStyleName:(NSString *)name
{
	NSString * loc = NSLocalizedString(name, @"");
    
	return loc;
}

- (NSString *) unlocalizedStyleName:(NSString *)name
{
	NSString * unloc = [fStyleLocMap objectForKey:name];
    
	return unloc ? unloc : name;
}

- (void)awakeFromNib
{
    [self loadVoiceMenu:fPrimaryVoiceMenu];
    [self loadVoiceMenu:fAlternateVoiceMenu];

	fStyleLocMap = [[NSMutableDictionary alloc] init];
    [fBoardStyle removeAllItems];
    [fPieceStyle removeAllItems];
    
	NSFileManager *	fileManager = [NSFileManager defaultManager];
	NSString	  * stylePath	= 
    [[[NSBundle mainBundle] resourcePath] 
     stringByAppendingPathComponent:@"Styles"];
	NSEnumerator  * styles 		= 
    [[fileManager contentsOfDirectoryAtPath:stylePath error:nil] objectEnumerator];
    
    BOOL excludeFurAndGrass = [MBCBoardWin isRenderingWithMetal];
	while (NSString * style = [styles nextObject]) {
        if (excludeFurAndGrass && ([style isEqualToString:@"Fur"] || [style isEqualToString:@"Grass"])) {
            continue;
        }
        
		NSString * locStyle = [self localizedStyleName:style];
		[fStyleLocMap setObject:style forKey:locStyle];
		NSString * s = [stylePath stringByAppendingPathComponent:style];
        if ([fileManager fileExistsAtPath:[s stringByAppendingPathComponent:@"Board.plist"]]) {
            [fBoardStyle addItemWithTitle:locStyle];
        }
        if ([fileManager fileExistsAtPath:[s stringByAppendingPathComponent:@"Piece.plist"]]) {
            [fPieceStyle addItemWithTitle:locStyle];
        }
	}
}

- (void) setPlayerAlias:(NSString *)playerID forKey:(NSString *)key
{
    if (!playerID)
        playerID = [fDocument nonLocalPlayerID];
    if (playerID)
        [GKPlayer loadPlayersForIdentifiers:[NSArray arrayWithObject:playerID]
            withCompletionHandler:^(NSArray *players, NSError *error) {
                if (!error) {
                    [fDocument setObject:[[players objectAtIndex:0] alias] forKey:key];
                    [self updateMoves:nil];
                }
            }];
}

- (void) removeChessObservers
{
    if (!fHasObservers)
        return;
    
    NSNotificationCenter * notificationCenter = [NSNotificationCenter defaultCenter];
    [notificationCenter removeObserver:self name:MBCEndMoveNotification object:nil];
    [notificationCenter removeObserver:self name:MBCTakebackNotification object:nil];
    [notificationCenter removeObserver:self name:MBCIllegalMoveNotification object:nil];
    [notificationCenter removeObserver:self name:MBCGameEndNotification object:nil];
    
    fHasObservers = NO;
}

- (void)dealloc
{
    [self removeChessObservers];
    [super dealloc];
}

- (void) startGame:(MBCVariant)variant playing:(MBCSide)sideToPlay
{
    NSLog(@"MBCGameInfo sideToPlay:%d", sideToPlay);
    [self removeChessObservers];
    NSNotificationCenter * notificationCenter = [NSNotificationCenter defaultCenter];
	[notificationCenter 
     addObserver:self
     selector:@selector(updateMoves:)
     name:MBCEndMoveNotification
     object:fDocument];
	[notificationCenter 
     addObserver:self
     selector:@selector(takeback:)
     name:MBCTakebackNotification
     object:fDocument];
	[notificationCenter 
     addObserver:self
     selector:@selector(updateMoves:)
     name:MBCIllegalMoveNotification
     object:fDocument];
	[notificationCenter 
     addObserver:self
     selector:@selector(gameEnd:)
     name:MBCGameEndNotification
     object:fDocument];
    fHasObservers = YES;

    //
    // Fill in missing properties
    //
    NSUserDefaults *        defaults    =   [NSUserDefaults standardUserDefaults];
    NSString *              human       =   [defaults stringForKey:kMBCHumanName];
    NSString *              human2      =   [defaults stringForKey:kMBCHumanName2];
    NSString *              engine      =   NSLocalizedString(@"engine_player", @"Computer");
    NSMutableDictionary *   props       =   fDocument.properties;
    BOOL                    gameCenter  =   [fDocument remoteSide] != kNeitherSide;
    
    [self setWhiteEditable: !gameCenter && SideIncludesWhite(sideToPlay)];
    [self setBlackEditable: !gameCenter && SideIncludesBlack(sideToPlay)];
    
    if (![props objectForKey:@"White"])
        if (gameCenter)
            [self setPlayerAlias:[props objectForKey:@"WhitePlayerID"] forKey:@"White"];
        else if (SideIncludesWhite(sideToPlay))
            [fDocument setObject:human forKey:@"White"];
        else 
            [fDocument setObject:engine forKey:@"White"];
    if (![props objectForKey:@"Black"])
        if (gameCenter)
            [self setPlayerAlias:[props objectForKey:@"BlackPlayerID"] forKey:@"Black"];
        else if (SideIncludesBlack(sideToPlay))
            [fDocument setObject:human2 forKey:@"Black"];
        else 
            [fDocument setObject:engine forKey:@"Black"];

    NSDate * now	= [NSDate date];
    if (![props objectForKey:@"StartDate"])
        [fDocument setObject:[now descriptionWithCalendarFormat:@"%Y.%m.%d" timeZone:nil locale:nil] 
                      forKey:@"StartDate"];
    if (![props objectForKey:@"StartTime"])
        [fDocument setObject:[now descriptionWithCalendarFormat:@"%H:%M:%S" timeZone:nil locale:nil] 
                      forKey:@"StartTime"];
    if (![props objectForKey:@"Result"])
        [fDocument setObject:@"*" forKey:@"Result"];
    if (![props objectForKey:@"City"])
        if (gameCenter)
            [fDocument setObject:NSLocalizedString(@"cloud_city", @"Game Center") forKey:@"City"];
        else
            [fDocument setObject:[defaults stringForKey:kMBCGameCity] forKey:@"City"];
    if (![props objectForKey:@"Country"])
        if (gameCenter)
            [fDocument setObject:NSLocalizedString(@"cloud_country", @"The Cloud") forKey:@"Country"];
        else
            [fDocument setObject:[defaults stringForKey:kMBCGameCountry] forKey:@"Country"];
    if (![props objectForKey:@"Event"])
        [fDocument setObject:[defaults stringForKey:kMBCGameEvent] forKey:@"Event"];

	fRows	= 0;
    
    [self updateMoves:nil];
}

- (void) updateMoves:(NSNotification *)notification
{
	[self willChangeValueForKey:@"gameTitle"];
    NSDictionary * props = fDocument.properties;
    if (![props objectForKey:@"White"])
        [self setPlayerAlias:[props objectForKey:@"WhitePlayerID"] forKey:@"White"];
    if (![props objectForKey:@"Black"])
        [self setPlayerAlias:[props objectForKey:@"BlackPlayerID"] forKey:@"Black"];
	[fMoveList reloadData];
	[fMoveList setNeedsDisplay:YES];

	int rows = [self numberOfRowsInTableView:fMoveList]; 
	if (rows != fRows) {
		fRows = rows;
		[fMoveList scrollRowToVisible:rows-1];
	}
	[self didChangeValueForKey:@"gameTitle"];
}

- (void) takeback:(NSNotification *)notification
{
    [fDocument setObject:@"*" forKey:@"Result"];

	[self updateMoves:notification];
}

- (void) gameEnd:(NSNotification *)notification
{
    dispatch_async(dispatch_get_main_queue(), ^{
        MBCMove *    move 	= reinterpret_cast<MBCMove *>([notification userInfo]);

        switch (move->fCommand) {
        case kCmdWhiteWins:
            [fDocument setObject:@"1-0" forKey:@"Result"];
            break;
        case kCmdBlackWins:
            [fDocument setObject:@"0-1" forKey:@"Result"];
            break;
        case kCmdDraw:
            [fDocument setObject:@"1/2-1/2" forKey:@"Result"];
            break;
        default:
            return;
        }
        [self updateMoves:notification];
    });
}

- (NSString *)outcome
{
    NSString * result = [fDocument objectForKey:@"Result"];
	if ([result isEqual:@"1-0"])
		return NSLocalizedString(@"white_win_msg", @"White wins");
	if ([result isEqual:@"0-1"])
		return NSLocalizedString(@"black_win_msg", @"Black wins");
	else if ([result isEqual:@"1/2-1/2"]) 
		return NSLocalizedString(@"draw_msg", @"Draw");
    
    return nil;
}

- (void)editProperties:(NSWindow *)sheet modalForWindow:(NSWindow *)window
{
    fEditedProperties = [[NSMutableDictionary alloc] init];
    
	[NSApp beginSheet:sheet
       modalForWindow:window
        modalDelegate:self
       didEndSelector:@selector(didEndSheet:returnCode:contextInfo:)
          contextInfo:nil];    
}

- (IBAction) editInfo:(id)sender
{
    NSLog(@"MBCGameInfo editInfo:");
    [self editProperties:fEditSheet modalForWindow:[sender window]];
}

- (void)editPreferencesForWindow:(NSWindow *)window hidePiecesStyle:(BOOL)hidePiecesStyle
{
    [fPrimaryVoiceMenu selectItemAtIndex:[self indexForVoice:[fDocument objectForKey:kMBCDefaultVoice]]];
    [fAlternateVoiceMenu selectItemAtIndex:[self indexForVoice:[fDocument objectForKey:kMBCAlternateVoice]]];
    [fBoardStyle selectItemWithTitle:[self localizedStyleName:[fDocument objectForKey:kMBCBoardStyle]]];
    [fPieceStyle selectItemWithTitle:[self localizedStyleName:[fDocument objectForKey:kMBCPieceStyle]]];
    
    if (hidePiecesStyle) {
        // Metal rendering unifies piece and board style to one popup, not text field labels needed for popups.
        fPieceStyle.hidden = YES;
        fPieceStyleText.hidden = YES;
        fPieceStyleText.stringValue = @"";
        fBoardStyleText.stringValue = @"";
        fPieceStyleTrailingConstraint.constant = 0;
        fBoardStyleTrailingConstraint.constant = 0;
    }
    
    [self editProperties:fPrefsSheet modalForWindow:window];
}

- (id)valueForUndefinedKey:(NSString *)key
{
    return [fDocument objectForKey:key];
}

- (void)setValue:(id)value forUndefinedKey:(NSString *)key
{
    if (![fEditedProperties objectForKey:key]) {
        id oldValue = [fDocument objectForKey:key];
        if (!oldValue)
            oldValue = [NSNull null];
        [fEditedProperties setObject:oldValue forKey:key];
    }

    [fDocument setValue:value forKey:key];
}

+ (NSSet *)keyPathsForValuesAffectingValueForKey:(NSString *)key
{
    if (isupper([key characterAtIndex:0]))
        return [NSSet setWithObject:[@"document." stringByAppendingString:key]];
    else 
        return [NSSet set];
}

- (void) didEndSheet:(NSWindow *)sheet returnCode:(NSInteger)returnCode contextInfo:(void *)ctx
{
    [sheet orderOut:self];
}

- (IBAction) cancelProperties:(id)sender
{
    //
    // Restore all the values that were changed
    //
	[NSApp endSheet:[sender window]];
    for (NSString * prop in fEditedProperties) 
        [fDocument setObject:[fEditedProperties objectForKey:prop] forKey:prop];
    [fEditedProperties release];
}

- (IBAction) updateProperties:(id)sender
{
	[self willChangeValueForKey:@"gameTitle"];
    NSUserDefaults * 	defaults 	= [NSUserDefaults standardUserDefaults];
   
    //
    // Update defaults
    //
    for (NSString * edited in fEditedProperties) {
        id val = [fDocument objectForKey:edited];
        if ([edited isEqual:@"White"]) { //|| ([edited isEqual:@"Black"] && ![fEditedProperties objectForKey:@"White"])) {
            [defaults setObject:val forKey:kMBCHumanName];
        }
        else if ([edited isEqual:@"Black"]) {
            [defaults setObject:val forKey:kMBCHumanName2];
        }
        else if ([edited isEqual:@"City"] ||[edited isEqual:@"Country"] || [edited isEqual:@"Event"])
            [defaults setObject:val forKey:[@"MBCGame" stringByAppendingString:edited]];
        else if ([[edited substringToIndex:3] isEqual:@"MBC"])
            [defaults setObject:val forKey:edited];
    }
    if ([fEditedProperties objectForKey:kMBCSearchTime])
        [fDocument updateSearchTime];

	[NSApp endSheet:[sender window]];
    [fEditedProperties release];
	[self didChangeValueForKey:@"gameTitle"];
}

- (IBAction) updateVoices:(id)sender;
{
    NSString * voice    = [self voiceAtIndex:[sender indexOfSelectedItem]];
    NSString * pvoice   = voice ? voice : @"";
    if ([sender tag])
        [self setValue:pvoice forKey:kMBCAlternateVoice];
    else
        [self setValue:pvoice forKey:kMBCDefaultVoice];

    NSSpeechSynthesizer *   selectedSynth = [[NSSpeechSynthesizer alloc] initWithVoice:voice];
    NSString *              demoText      = 
        [[NSSpeechSynthesizer attributesForVoice:[selectedSynth voice]] 
         objectForKey:NSVoiceDemoText];
    if (demoText)
        [selectedSynth startSpeakingString:demoText];
}

- (IBAction) updateStyles:(id)sender;
{
	NSString *			boardStyle	= 
        [self unlocalizedStyleName:[fBoardStyle titleOfSelectedItem]];
	NSString *			pieceStyle  = 
        [self unlocalizedStyleName:[fPieceStyle titleOfSelectedItem]];

	[self setValue:boardStyle forKey:kMBCBoardStyle];
	[self setValue:pieceStyle forKey:kMBCPieceStyle];
}

//
//     If there is no document anymore, fBoard can't be trusted either
//
- (void)setDocument:(MBCDocument *)document
{
    fDocument = document;
    if (!fDocument)
        fBoard = nil;
}

- (int)numberOfRowsInTableView:(NSTableView *)aTableView
{
	return ([fBoard numMoves]+1) / 2;
}

- (id)tableView:(NSTableView *)v objectValueForTableColumn:(NSTableColumn *)col row:(int)row
{
	NSString * 		ident 	= [col identifier];
	if ([ident isEqual:@"Move"]) {
		return [NSNumber numberWithInt:row+1];
	} else {
		NSArray * identComp = [ident componentsSeparatedByString:@"."];
		return [[fBoard move: row*2+[[identComp objectAtIndex:0] isEqual:@"Black"]] 
				valueForKey:[identComp objectAtIndex:1]];
	}
    return nil;
}

- (BOOL)tableView:(NSTableView *)aTableView shouldSelectRow:(NSInteger)rowIndex
{
	return NO; /* Disallow all selections */
}

- (NSString *)describeMove:(int)move
{
    NSDictionary * localization = nil;
    
    if (NSURL * url = [[NSBundle mainBundle] URLForResource:@"Spoken" withExtension:@"strings"
                                    subdirectory:nil]
    )
        localization = [NSDictionary dictionaryWithContentsOfURL:url];
    NSString * nr_fmt =
        NSLocalizedStringFromTable(@"move_table_nr", @"Spoken", @"Move %d");
    NSString * text = [NSString localizedStringWithFormat:nr_fmt, move];
    int white = (move-1)*2;
    if (white < [fBoard numMoves])
        text = [text stringByAppendingFormat:@"\n\n%@",
                [fBoard extStringFromMove:[fBoard move:white]
                         withLocalization:localization]];
    int black = move*2 - 1;
    if (black < [fBoard numMoves])
        text = [text stringByAppendingFormat:@"\n\n%@",
                [fBoard extStringFromMove:[fBoard move:black]
                         withLocalization:localization]];
    
    return text;
}

- (NSString *)gameTitle
{
    if (!fDocument || [fDocument brandNewGame])
        return @"";
    
	NSString * 		move;
	int		 		numMoves = [fBoard numMoves];		
	
	if (numMoves && [[NSUserDefaults standardUserDefaults] boolForKey:kMBCShowMoveInTitle]) {
		NSNumber * moveNum = [NSNumber numberWithInt:(numMoves+1)/2];
		NSString * moveStr = [NSNumberFormatter localizedStringFromNumber:moveNum 
															  numberStyle:NSNumberFormatterDecimalStyle];
		move = 	[NSString localizedStringWithFormat:NSLocalizedString(@"title_move_line_fmt", @"%@. %@%@"),
						  moveStr, numMoves&1 ? @"":@"\xE2\x80\xA6 ",
						  [[fBoard lastMove] localizedText]];
    } else if ([[fDocument objectForKey:@"Request"] isEqual:@"Takeback"]) {
        move = NSLocalizedString(@"takeback_msg", @"Takeback requested");
	} else if (numMoves & 1) {
		move = 	NSLocalizedString(@"black_move_msg", @"Black to move");
	} else {
		move = 	NSLocalizedString(@"white_move_msg", @"White to move");
	}

	if (NSString * outcome = [self outcome])
		move = outcome;
	NSString * title =
    [NSString localizedStringWithFormat:NSLocalizedString(@"game_title_fmt", @"%@ : %@ - %@   (%@)"),
                fDocument.displayName,
                [fDocument objectForKey:@"White"],
                [fDocument objectForKey:@"Black"], move];
    
    return title;
}

@end

// Local Variables:
// mode:ObjC
// End:
