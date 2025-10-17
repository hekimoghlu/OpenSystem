/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 24, 2022.
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


#import "MockSynchronousEscrowServer.h"

@interface MockSynchronousEscrowServer ()
@property EscrowRequestServer* server;
@end

@implementation MockSynchronousEscrowServer

- (instancetype)initWithServer:(EscrowRequestServer*)server
{
    if((self = [super init])) {
        _server = server;
    }
    return self;
}

- (void)cachePrerecord:(NSString*)uuid
   serializedPrerecord:(nonnull NSData *)prerecord
                 reply:(nonnull void (^)(NSError * _Nullable))reply
{
    dispatch_semaphore_t sema = dispatch_semaphore_create(0);

    [self.server cachePrerecord:uuid
            serializedPrerecord:prerecord
                          reply:^(NSError * _Nullable error) {
                              reply(error);
                              dispatch_semaphore_signal(sema);
                          }];

    dispatch_semaphore_wait(sema, DISPATCH_TIME_FOREVER);
}

- (void)fetchPrerecord:(nonnull NSString *)prerecordUUID
                 reply:(nonnull void (^)(NSData * _Nullable, NSError * _Nullable))reply
{

    dispatch_semaphore_t sema = dispatch_semaphore_create(0);

    [self.server fetchPrerecord:prerecordUUID
                          reply:^(NSData* contents, NSError * _Nullable error) {
                              reply(contents, error);
                              dispatch_semaphore_signal(sema);
                          }];

    dispatch_semaphore_wait(sema, DISPATCH_TIME_FOREVER);
}

- (void)fetchRequestWaitingOnPasscode:(nonnull void (^)(NSString * _Nullable, NSError * _Nullable))reply
{
    dispatch_semaphore_t sema = dispatch_semaphore_create(0);

    [self.server fetchRequestWaitingOnPasscode:^(NSString* uuid, NSError * _Nullable error) {
                              reply(uuid, error);
                              dispatch_semaphore_signal(sema);
                          }];

    dispatch_semaphore_wait(sema, DISPATCH_TIME_FOREVER);
}

- (void)triggerEscrowUpdate:(nonnull NSString *)reason
                      reply:(nonnull void (^)(NSError * _Nullable))reply
{
    dispatch_semaphore_t sema = dispatch_semaphore_create(0);

    [self.server triggerEscrowUpdate:reason reply:^(NSError * _Nullable error) {
        reply(error);
        dispatch_semaphore_signal(sema);
    }];

    dispatch_semaphore_wait(sema, DISPATCH_TIME_FOREVER);
}

- (void)fetchRequestStatuses:(nonnull void (^)(NSDictionary<NSString *,NSString *> * _Nullable, NSError * _Nullable))reply {
    dispatch_semaphore_t sema = dispatch_semaphore_create(0);

    [self.server fetchRequestStatuses:^(NSDictionary<NSString *,NSString *> * dict, NSError * _Nullable error) {
        reply(dict, error);
        dispatch_semaphore_signal(sema);
    }];

    dispatch_semaphore_wait(sema, DISPATCH_TIME_FOREVER);
}

- (void)resetAllRequests:(nonnull void (^)(NSError * _Nullable))reply {
    dispatch_semaphore_t sema = dispatch_semaphore_create(0);

    [self.server resetAllRequests:^(NSError * _Nullable error) {
        reply(error);
        dispatch_semaphore_signal(sema);
    }];

    dispatch_semaphore_wait(sema, DISPATCH_TIME_FOREVER);
}

- (void)storePrerecordsInEscrow:(nonnull void (^)(uint64_t, NSError * _Nullable))reply {

    dispatch_semaphore_t sema = dispatch_semaphore_create(0);

    [self.server storePrerecordsInEscrow:^(uint64_t x, NSError * _Nullable error) {
        reply(x, error);
        dispatch_semaphore_signal(sema);
    }];

    dispatch_semaphore_wait(sema, DISPATCH_TIME_FOREVER);
}

@end
