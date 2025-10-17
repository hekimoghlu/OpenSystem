/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 11, 2024.
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
#import "KDAppDelegate.h"
#import "KDCirclePeer.h"
#import "NSArray+mapWithBlock.h"
#include <notify.h>

#include <Security/SecItemInternal.h>

@implementation KDAppDelegate

- (void)applicationDidFinishLaunching:(NSNotification *)aNotification
{
    self.stuffNotToLeak = [NSMutableArray new];
    [self.stuffNotToLeak addObject:[[NSNotificationCenter defaultCenter] addObserverForName:kKDSecItemsUpdated object:nil queue:[NSOperationQueue mainQueue] usingBlock:^(NSNotification *note) {
        self.itemTableTitle.title = [NSString stringWithFormat:@"All Items (%ld)", (long)[self.itemDataSource numberOfRowsInTableView:self.itemTable]];
    }]];
    
    [self.syncSpinner setUsesThreadedAnimation:YES];
    [self.syncSpinner startAnimation:nil];
    
    self.itemDataSource = [[KDSecItems alloc] init];
    self.itemTable.dataSource = self.itemDataSource;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
    int notificationToken;
    uint32_t rc = notify_register_dispatch(kSecServerKeychainChangedNotification, &notificationToken, dispatch_get_main_queue(), ^(int token __unused) {
            NSLog(@"Received %s", kSecServerKeychainChangedNotification);
            [(KDSecItems*)self.itemDataSource loadItems];
            [self.itemTable reloadData];
         });
    NSAssert(rc == 0, @"Can't register for %s", kSecServerKeychainChangedNotification);
#pragma clang diagnostic pop
	
	self.circle = [KDSecCircle new];

    __weak typeof(self) weakSelf = self;
	[self.circle addChangeCallback:^{
        __strong typeof(self) strongSelf = weakSelf;
        if(strongSelf) {
            strongSelf.circleStatusCell.stringValue = strongSelf.circle.status;

            [strongSelf setCheckbox];

            strongSelf.peerCountCell.objectValue = @(strongSelf.circle.peers.count);
            NSString *peerNames = [[strongSelf.circle.peers mapWithBlock:^id(id obj) {
                return ((KDCirclePeer*)obj).name;
            }] componentsJoinedByString:@"\n"];
            [strongSelf.peerTextList.textStorage replaceCharactersInRange:NSMakeRange(0, [strongSelf.peerTextList.textStorage length]) withString:peerNames];

            strongSelf.applicantCountCell.objectValue = @(strongSelf.circle.applicants.count);
            NSString *applicantNames = [[strongSelf.circle.applicants mapWithBlock:^id(id obj) {
                return ((KDCirclePeer*)obj).name;
            }] componentsJoinedByString:@"\n"];
            [strongSelf.applicantTextList.textStorage replaceCharactersInRange:NSMakeRange(0, [strongSelf.applicantTextList.textStorage length]) withString:applicantNames];

            [strongSelf.syncSpinner stopAnimation:nil];
        }
	}];
}

-(void)setCheckbox
{
    if (self.circle.isInCircle) {
        [self.enableKeychainSyncing setState:NSControlStateValueOn];
    } else if (self.circle.isOutOfCircle) {
        [self.enableKeychainSyncing setState:NSControlStateValueOff];
    } else {
        [self.enableKeychainSyncing setState:NSControlStateValueMixed];
    }
}

-(IBAction)enableKeychainSyncingClicked:(id)sender
{
    [self.syncSpinner startAnimation:sender];
    if (self.circle.isOutOfCircle) {
        [self.circle enableSync];
    } else {
        [self.circle disableSync];
    }
    [self setCheckbox];
}

@end
