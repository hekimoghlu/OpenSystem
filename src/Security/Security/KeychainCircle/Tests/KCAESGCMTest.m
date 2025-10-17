/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 22, 2021.
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

//
//  KCAESGCMTest.m
//  Keychain Circle
//
//

#import <XCTest/XCTest.h>
#import <XCTest/XCTestCase_Private.h>

#import <Foundation/Foundation.h>
#import <KeychainCircle/KCAESGCMDuplexSession.h>
#import <Foundation/NSKeyedArchiver_Private.h>

@interface KCAESGCMTest : XCTestCase

@end

@implementation KCAESGCMTest

- (void) sendMessage: (NSData*) message
                from: (KCAESGCMDuplexSession*) sender
                  to: (KCAESGCMDuplexSession*) receiver {
    NSError* error = nil;
    NSData* sendToRecv = [sender encrypt:message error:&error];

    XCTAssertNil(error, @"Got error");
    XCTAssertNotNil(sendToRecv, @"Failed to get data");

    error = nil;
    NSData* decryptedSendToRecv = [receiver decryptAndVerify:sendToRecv error:&error];

    XCTAssertNil(error, @"Error decrypting");
    XCTAssertNotNil(decryptedSendToRecv, @"Got decryption");

    XCTAssertEqualObjects(message, decryptedSendToRecv, @"Send to recv failed.");
}

- (void)testAESGCMDuplex {
#if XCT_MEMORY_TESTING_AVAILABLE
    [self assertNoLeaksInScope:^{
#endif /* XCT_MEMORY_TESTING_AVAILABLE */
        uint64_t context = 0x81FC134000123041;
        uint8_t secretBytes[] = { 0x11, 0x22, 0x33, 0x13, 0x44, 0xF1, 0x13, 0x92, 0x11, 0x22, 0x33, 0x13, 0x44, 0xF1, 0x13, 0x92 };
        NSData* secret = [NSData dataWithBytes:secretBytes length:sizeof(secretBytes)];

        KCAESGCMDuplexSession* sender = [KCAESGCMDuplexSession sessionAsSender:secret
                                                                       context:context];

        KCAESGCMDuplexSession* receiver = [KCAESGCMDuplexSession sessionAsReceiver:secret
                                                                           context:context];

        uint8_t sendToRecvBuffer[] = { 0x1, 0x2, 0x3, 0x88, 0xFF, 0xE1 };
        NSData* sendToRecvData = [NSData dataWithBytes:sendToRecvBuffer length:sizeof(sendToRecvBuffer)];

        [self sendMessage:sendToRecvData from:sender to:receiver];

        uint8_t recvToSendBuffer[] = { 0x81, 0x52, 0x63, 0x88, 0xFF, 0xE1 };
        NSData* recvToSendData = [NSData dataWithBytes:recvToSendBuffer length:sizeof(recvToSendBuffer)];

        [self sendMessage:recvToSendData from:receiver to:sender];
#if XCT_MEMORY_TESTING_AVAILABLE
    }];
#endif /* XCT_MEMORY_TESTING_AVAILABLE */
}

- (KCAESGCMDuplexSession*) archiveDearchive: (KCAESGCMDuplexSession*) original {
    NSKeyedArchiver *archiver = [[NSKeyedArchiver alloc] initRequiringSecureCoding:YES];
    [archiver encodeObject:original forKey:@"Top"];
    [archiver finishEncoding];


    // Customize the unarchiver.
    NSKeyedUnarchiver *unarchiver = [[NSKeyedUnarchiver alloc] initForReadingFromData:archiver.encodedData error:nil];
    KCAESGCMDuplexSession *result = [unarchiver decodeObjectOfClass:[KCAESGCMDuplexSession class] forKey:@"Top"];
    [unarchiver finishDecoding];

    return result;
}

- (void)doAESGCMDuplexCodingFlattenSender: (bool) flattenSender
                                 Receiver: (bool) flattenReceiver {
    uint64_t context = 0x81FC134000123041;
    uint8_t secretBytes[] = { 0x73, 0xb7, 0x7f, 0xff, 0x7f, 0xe3, 0x44, 0x6b, 0xa4, 0xec, 0x9d, 0x5d, 0x68, 0x12, 0x13, 0x71 };
    NSData* secret = [NSData dataWithBytes:secretBytes length:sizeof(secretBytes)];

    KCAESGCMDuplexSession* sender = [KCAESGCMDuplexSession sessionAsSender:secret
                                                                   context:context];

    KCAESGCMDuplexSession* receiver = [KCAESGCMDuplexSession sessionAsReceiver:secret
                                                                       context:context];

    {
        uint8_t sendToRecvBuffer[] = { 0x0e, 0x9b, 0x9d, 0x2c, 0x90, 0x96, 0x8a };
        NSData* sendToRecvData = [NSData dataWithBytes:sendToRecvBuffer length:sizeof(sendToRecvBuffer)];

        [self sendMessage:sendToRecvData from:sender to:receiver];


        uint8_t recvToSendBuffer[] = {  0x9b, 0x63, 0xaf, 0xb5, 0x4d, 0xa0, 0xfa, 0x9d, 0x90 };
        NSData* recvToSendData = [NSData dataWithBytes:recvToSendBuffer length:sizeof(recvToSendBuffer)];

        [self sendMessage:recvToSendData from:receiver to:sender];
    }

    // Re-encode...
    if (flattenSender) {
        sender = [self archiveDearchive:sender];
    }

    if (flattenReceiver) {
        receiver = [self archiveDearchive:receiver];
    }

    {
        uint8_t sendToRecvBuffer[] = { 0xae, 0xee, 0x5f, 0x62, 0xb2, 0x72, 0x6f, 0x0a, 0xb6, 0x56 };
        NSData* sendToRecvData = [NSData dataWithBytes:sendToRecvBuffer length:sizeof(sendToRecvBuffer)];

        [self sendMessage:sendToRecvData from:sender to:receiver];


        uint8_t recvToSendBuffer[] = { 0x49, 0x0b, 0xbb, 0x2d, 0x20, 0xb1, 0x8a, 0xfc, 0xba, 0xd1, 0xFF };
        NSData* recvToSendData = [NSData dataWithBytes:recvToSendBuffer length:sizeof(recvToSendBuffer)];

        [self sendMessage:recvToSendData from:receiver to:sender];
    }
}

- (void)testAESGCMDuplexCodingFlattenReceiver {
#if XCT_MEMORY_TESTING_AVAILABLE
    [self assertNoLeaksInScope:^{
#endif /* XCT_MEMORY_TESTING_AVAILABLE */
        [self doAESGCMDuplexCodingFlattenSender:NO Receiver:YES];
#if XCT_MEMORY_TESTING_AVAILABLE
    }];
#endif /* XCT_MEMORY_TESTING_AVAILABLE */
}

- (void)testAESGCMDuplexCodingFlattenSender {
#if XCT_MEMORY_TESTING_AVAILABLE
    [self assertNoLeaksInScope:^{
#endif /* XCT_MEMORY_TESTING_AVAILABLE */
        [self doAESGCMDuplexCodingFlattenSender:YES Receiver:NO];
#if XCT_MEMORY_TESTING_AVAILABLE
    }];
#endif /* XCT_MEMORY_TESTING_AVAILABLE */
}

- (void)testAESGCMDuplexCodingFlattenSenderReceiver {
#if XCT_MEMORY_TESTING_AVAILABLE
    [self assertNoLeaksInScope:^{
#endif /* XCT_MEMORY_TESTING_AVAILABLE */
        [self doAESGCMDuplexCodingFlattenSender:YES Receiver:YES];
#if XCT_MEMORY_TESTING_AVAILABLE
    }];
#endif /* XCT_MEMORY_TESTING_AVAILABLE */
}


@end
