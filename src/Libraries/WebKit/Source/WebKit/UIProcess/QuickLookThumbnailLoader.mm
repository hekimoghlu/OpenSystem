/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 6, 2024.
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
#import "config.h"
#import "QuickLookThumbnailLoader.h"

#if HAVE(QUICKLOOK_THUMBNAILING)

#import "APIAttachment.h"
#import <wtf/FileSystem.h>
#import "QuickLookThumbnailingSoftLink.h"

@implementation WKQLThumbnailQueueManager {
    RetainPtr<NSOperationQueue> _queue;
}

- (instancetype)init
{
    self = [super init];
    if (self)
        _queue = adoptNS([[NSOperationQueue alloc] init]);
    return self;
}

- (void)dealloc
{
    [super dealloc];
}

- (NSOperationQueue *)queue
{
    return _queue.get();
}

+ (WKQLThumbnailQueueManager *)sharedInstance
{
    static WKQLThumbnailQueueManager *sharedInstance = [[WKQLThumbnailQueueManager alloc] init];
    return sharedInstance;
}

@end

@interface WKQLThumbnailLoadOperation ()
@property (atomic, readwrite, getter=isExecuting) BOOL executing;
@property (atomic, readwrite, getter=isFinished) BOOL finished;
@end

@implementation WKQLThumbnailLoadOperation {
    RetainPtr<NSURL> _filePath;
    RetainPtr<NSString> _identifier;
    RefPtr<const API::Attachment> _attachment;
    RetainPtr<CocoaImage> _thumbnail;
    BOOL _shouldWrite;
}

- (instancetype)initWithAttachment:(const API::Attachment&)attachment identifier:(NSString *)identifier
{
    if (self = [super init]) {
        _attachment = &attachment;
        _identifier = adoptNS([identifier copy]);
        _shouldWrite = YES;
    }
    return self;
}

- (instancetype)initWithURL:(NSString *)fileURL identifier:(NSString *)identifier
{
    if (self = [super init]) {
        _identifier = adoptNS([identifier copy]);
        _filePath = [NSURL fileURLWithPath:fileURL];
    }
    return self;
}

- (void)start
{
    self.executing = YES;

    if (_shouldWrite) {
        NSString *temporaryDirectory = FileSystem::createTemporaryDirectory(@"QLTempFileData");

        NSFileWrapperWritingOptions options = 0;
        NSError *error = nil;

        _attachment->doWithFileWrapper([&](NSFileWrapper *fileWrapper) {
            auto fileURLPath = [NSURL fileURLWithPath:[temporaryDirectory stringByAppendingPathComponent:fileWrapper.preferredFilename]];
            [fileWrapper writeToURL:fileURLPath options:options originalContentsURL:nil error:&error];
            _filePath = WTFMove(fileURLPath);
        });
        if (error)
            return;
    }

    auto request = adoptNS([WebKit::allocQLThumbnailGenerationRequestInstance() initWithFileAtURL:_filePath.get() size:CGSizeMake(400, 400) scale:1 representationTypes:QLThumbnailGenerationRequestRepresentationTypeThumbnail]);
    [request setIconMode:YES];
    
    [[WebKit::getQLThumbnailGeneratorClass() sharedGenerator] generateBestRepresentationForRequest:request.get() completionHandler:^(QLThumbnailRepresentation *thumbnail, NSError *error) {
        if (error)
            return;
        if (_thumbnail)
            return;
#if USE(APPKIT)
        _thumbnail = thumbnail.NSImage;
#else
        _thumbnail = thumbnail.UIImage;
#endif
        if (_shouldWrite)
            [[NSFileManager defaultManager] removeItemAtURL:_filePath.get() error:nullptr];

        self.executing = NO;
        self.finished = YES;
    }];
}

- (CocoaImage *)thumbnail
{
    return _thumbnail.get();
}

- (NSString *)identifier
{
    return _identifier.get();
}

- (BOOL)isAsynchronous
{
    return YES;
}

@synthesize executing = _executing;

- (BOOL)isExecuting
{
    @synchronized(self) {
        return _executing;
    }
}

- (void)setExecuting:(BOOL)executing
{
    @synchronized(self) {
        if (executing != _executing) {
            [self willChangeValueForKey:@"isExecuting"];
            _executing = executing;
            [self didChangeValueForKey:@"isExecuting"];
        }
    }
}

@synthesize finished = _finished;

- (BOOL)isFinished
{
    @synchronized(self) {
        return _finished;
    }
}

- (void)setFinished:(BOOL)finished
{
    @synchronized(self) {
        if (finished != _finished) {
            [self willChangeValueForKey:@"isFinished"];
            _finished = finished;
            [self didChangeValueForKey:@"isFinished"];
        }
    }
}

@end

#endif // HAVE(QUICKLOOK_THUMBNAILING)
