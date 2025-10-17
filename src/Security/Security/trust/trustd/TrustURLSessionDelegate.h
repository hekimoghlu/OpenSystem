/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 2, 2023.
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
#ifndef _SECURITY_TRUSTURLSESSIONDELEGATE_H_
#define _SECURITY_TRUSTURLSESSIONDELEGATE_H_

#if __OBJC__
#include <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface TrustURLSessionContext : NSObject
@property (assign, nullable) void *context;
@property NSArray <NSURL *>*URIs;
@property NSUInteger URIix;
@property (nullable) NSMutableData *response;
@property NSTimeInterval maxAge;
@property NSUInteger numTasks;
@property NSURLRequestAttribution attribution;

- (instancetype)initWithContext:(CFTypeRef)context uris:(NSArray <NSURL *>*)uris;
@end

@interface NSURLRequest (TrustURLRequest)
- (NSUUID * _Nullable)taskId;
@end

/* This is our abstract NSURLSessionDelegate that handles the elements common to
 * fetching data over the network during a trust evaluation */
@interface TrustURLSessionDelegate : NSObject <NSURLSessionDelegate, NSURLSessionTaskDelegate, NSURLSessionDataDelegate>
@property dispatch_queue_t queue;

/* The delegate superclass keeps track of all the tasks that have been kicked off (via fetchNext);
 * it is the responsibility of the subclass to remove tasks when it is done with them. */
- (TrustURLSessionContext *)contextForTask:(NSUUID *)taskId;
- (void)removeTask:(NSUUID *)taskId;

- (BOOL)fetchNext:(NSURLSession *)session context:(TrustURLSessionContext *)context;
- (NSURLRequest *)createNextRequest:(NSURL *)uri context:(TrustURLSessionContext *)context;
@end

NS_ASSUME_NONNULL_END
#endif // __OBJC__

#endif /* _SECURITY_TRUSTURLSESSIONDELEGATE_H_ */
