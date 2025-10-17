/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 13, 2023.
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
#import "KDSecCircle.h"
#import "KDCirclePeer.h"
#include <notify.h>
#include <dispatch/dispatch.h>

#import <Security/SecureObjectSync/SOSCloudCircle.h>
#import <Security/SecureObjectSync/SOSPeerInfo.h>

#import <CloudServices/SecureBackup.h>

#include <utilities/debugging.h>

@interface KDSecCircle ()
@property (retain) NSMutableArray *callbacks;

@property (readwrite) unsigned long long changeCount;

@property (readwrite) SOSCCStatus rawStatus;

@property (readwrite) NSString *status;
@property (readwrite) NSError *error;

@property (readwrite) NSArray *peers;
@property (readwrite) NSArray *applicants;

@property (readwrite) dispatch_queue_t queue_;

@end

@implementation KDSecCircle

-(void)updateCheck
{
	// XXX: assert not on main_queue
	CFErrorRef err = NULL;

    SOSCCValidateUserPublic(NULL); // requires the account queue - makes the rest of this wait for fresh info.  This used to happen in SOSCCThisDeviceIsInCircle(below) before we made it use cached info.
	SOSCCStatus newRawStatus = SOSCCThisDeviceIsInCircle(&err);
    NSArray *peerInfos = (__bridge_transfer NSArray *) SOSCCCopyApplicantPeerInfo(&err);
    NSMutableArray *newApplicants = [[NSMutableArray alloc] initWithCapacity:peerInfos.count];
	[peerInfos enumerateObjectsUsingBlock:^(id obj, NSUInteger idx, BOOL *stop) {
        [newApplicants addObject:[[KDCirclePeer alloc] initWithPeerObject:obj]];
    }];
	
	peerInfos = (__bridge_transfer NSArray *) SOSCCCopyPeerPeerInfo(&err);
    NSMutableArray *newPeers = [[NSMutableArray alloc] initWithCapacity:peerInfos.count];
	[peerInfos enumerateObjectsUsingBlock:^(id obj, NSUInteger idx, BOOL *stop) {
        [newPeers addObject:[[KDCirclePeer alloc] initWithPeerObject:obj]];
    }];
    
    secdebug("kcn", "rawStatus %d, #applicants %lu, #peers %lu, err=%@", newRawStatus, (unsigned long)[newApplicants count], (unsigned long)[newPeers count], err);

	dispatch_async(dispatch_get_main_queue(), ^{
        self.rawStatus = newRawStatus;
        
		switch (newRawStatus) {
			case kSOSCCInCircle:
				self.status = @"In Circle";
				break;
				
			case kSOSCCNotInCircle:
				self.status = @"Not In Circle";
				break;

			case kSOSCCRequestPending:
				self.status = @"Request Pending";
				break;
				
			case kSOSCCCircleAbsent:
				self.status = @"Circle Absent";
				break;
				
			case kSOSCCError:
				self.status = [NSString stringWithFormat:@"Error: %@", err];
				break;
				
			default:
				self.status = [NSString stringWithFormat:@"Unknown status code %d", self.rawStatus];
				break;
		}
        
        self.applicants = [newApplicants copy];
        self.peers      = [newPeers copy];
        self.error      = (__bridge NSError *)(err);

        self.changeCount++;
        for (dispatch_block_t callback in self.callbacks) {
            callback();
        }
	});
}

// XXX It's a botch to use the "name" and not applicant, but
// it is hard to get anythign else to survive a serialastion
// trip thoguth NSUserNotificationCenter.
//
// Er, now that I look more closely maybe SOSPeerInfoGetPeerID...

typedef void (^applicantBlock)(id applicant);

-(void)forApplicantId:(NSString*)applicantId run:(applicantBlock)applicantBlock
{
    dispatch_async(self.queue_, ^{
		for (KDCirclePeer *applicant in self.applicants) {
			if ([applicantId isEqualToString:applicantId]) {
				applicantBlock(applicant.peerObject);
				break;
			}
		}
    });
}

// Tell clang that these bools are okay, even if NSAssert doesn't use them
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"

-(void)acceptApplicantId:(NSString*)applicantId
{
    [self forApplicantId:applicantId run:^void(id applicant) {
        CFErrorRef err = NULL;
        bool ok = SOSCCAcceptApplicants((__bridge CFArrayRef)(@[applicant]), &err);
        NSAssert(ok, @"Error %@ while accepting %@ (%@)", err, applicantId, applicant);
    }];
}

-(void)rejectApplicantId:(NSString*)applicantId
{
    [self forApplicantId:applicantId run:^void(id applicant) {
        CFErrorRef err = NULL;
        bool ok = SOSCCRejectApplicants((__bridge CFArrayRef)(@[applicant]), &err);
        NSAssert(ok, @"Error %@ while rejecting %@ (%@)", err, applicantId, applicant);
    }];
}

#pragma clang diagnostic pop

-(id)init
{
    if ((self = [super init])) {
        int token;
    
        self->_queue_ = dispatch_queue_create([[NSString stringWithFormat:@"KDSecCircle@%p", self] UTF8String], NULL);
        self->_callbacks = [NSMutableArray new];
        notify_register_dispatch(kSOSCCCircleChangedNotification, &token, self.queue_, ^(int token1) {
                [self updateCheck];
            });
    }
    return self;
}

-(void)addChangeCallback:(dispatch_block_t)callback
{
	[self.callbacks addObject:callback];
    if (self.changeCount) {
        dispatch_async(dispatch_get_main_queue(), callback);
    } else if (self.callbacks.count == 1) {
        dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
            [self updateCheck];
        });
    }
}

-(BOOL)isInCircle
{
    return (self.rawStatus == kSOSCCInCircle);
}

-(BOOL)isOutOfCircle
{
    return (self.rawStatus == kSOSCCNotInCircle || self.rawStatus == kSOSCCCircleAbsent);
}

-(void)enableSync
{
    CFErrorRef err = NULL;
	if (self.rawStatus == kSOSCCCircleAbsent) {
		SOSCCResetToOffering(&err);
	} else {
		SOSCCRequestToJoinCircle(&err);
	}
    
    CFMutableSetRef viewsToEnable = CFSetCreateMutable(NULL, 0, NULL);
    CFMutableSetRef viewsToDisable = CFSetCreateMutable(NULL, 0, NULL);
    CFSetAddValue(viewsToEnable, (void*)kSOSViewWiFi);
    CFSetAddValue(viewsToEnable, (void*)kSOSViewAutofillPasswords);
    CFSetAddValue(viewsToEnable, (void*)kSOSViewSafariCreditCards);
    CFSetAddValue(viewsToEnable, (void*)kSOSViewOtherSyncable);
    
    SOSCCViewSet(viewsToEnable, viewsToDisable);
    CFRelease(viewsToEnable);
    CFRelease(viewsToDisable);
}

-(void)disableSync
{
    CFErrorRef err = NULL;
    SOSCCRemoveThisDeviceFromCircle(&err);
}

@end
