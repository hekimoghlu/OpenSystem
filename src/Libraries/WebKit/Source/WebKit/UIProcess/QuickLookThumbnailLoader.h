/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 1, 2022.
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
#if HAVE(QUICKLOOK_THUMBNAILING)

#import "CocoaImage.h"

namespace API {
class Attachment;
}

@interface WKQLThumbnailQueueManager : NSObject

@property (nonatomic, readonly) NSOperationQueue *queue;

- (instancetype)init;
+ (WKQLThumbnailQueueManager *)sharedInstance;

@end

@interface WKQLThumbnailLoadOperation : NSOperation

@property (atomic, readonly, getter=isAsynchronous) BOOL asynchronous;
@property (atomic, readonly, getter=isExecuting) BOOL executing;
@property (atomic, readonly, getter=isFinished) BOOL finished;

@property (nonatomic, readonly, copy) NSString *identifier;
@property (nonatomic, readonly, retain) CocoaImage *thumbnail;

- (instancetype)initWithAttachment:(const API::Attachment&)attachment identifier:(NSString *)identifier;
- (instancetype)initWithURL:(NSString *)fileURL identifier:(NSString *)identifier;

@end

#endif // HAVE(QUICKLOOK_THUMBNAILING)
