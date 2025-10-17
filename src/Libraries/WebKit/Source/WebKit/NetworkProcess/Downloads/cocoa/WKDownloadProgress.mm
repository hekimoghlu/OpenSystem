/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 5, 2024.
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
#import "WKDownloadProgress.h"

#import "Download.h"
#import <pal/spi/cocoa/NSProgressSPI.h>
#import <sys/xattr.h>
#import <wtf/BlockPtr.h>
#import <wtf/WeakObjCPtr.h>

static void* WKDownloadProgressBytesExpectedToReceiveCountContext = &WKDownloadProgressBytesExpectedToReceiveCountContext;
static void* WKDownloadProgressBytesReceivedContext = &WKDownloadProgressBytesReceivedContext;

static NSString * const countOfBytesExpectedToReceiveKeyPath = @"countOfBytesExpectedToReceive";
static NSString * const countOfBytesReceivedKeyPath = @"countOfBytesReceived";

#if HAVE(MODERN_DOWNLOADPROGRESS)
#import <WebKitAdditions/DownloadProgressAdditions.mm>
#endif

@implementation WKDownloadProgress {
    RetainPtr<NSURLSessionDownloadTask> m_task;
    WeakPtr<WebKit::Download> m_download;
    RefPtr<WebKit::SandboxExtension> m_sandboxExtension;
}

- (void)performCancel
{
    if (m_download)
        m_download->cancel([](auto) { }, WebKit::Download::IgnoreDidFailCallback::No);
    m_download = nullptr;
}

- (instancetype)initWithDownloadTask:(NSURLSessionDownloadTask *)task download:(WebKit::Download&)download URL:(NSURL *)fileURL sandboxExtension:(RefPtr<WebKit::SandboxExtension>)sandboxExtension
{
    if (!(self = [self initWithParent:nil userInfo:nil]))
        return nil;

    m_task = task;
    m_download = download;

    [task addObserver:self forKeyPath:countOfBytesExpectedToReceiveKeyPath options:NSKeyValueObservingOptionNew | NSKeyValueObservingOptionInitial context:WKDownloadProgressBytesExpectedToReceiveCountContext];
    [task addObserver:self forKeyPath:countOfBytesReceivedKeyPath options:NSKeyValueObservingOptionNew | NSKeyValueObservingOptionInitial context:WKDownloadProgressBytesReceivedContext];

    self.kind = NSProgressKindFile;
    self.fileOperationKind = NSProgressFileOperationKindDownloading;
    self.fileURL = fileURL;
    m_sandboxExtension = sandboxExtension;

    self.cancellable = YES;
    self.cancellationHandler = makeBlockPtr([weakSelf = WeakObjCPtr<WKDownloadProgress> { self }] () mutable {
        ensureOnMainRunLoop([weakSelf = WTFMove(weakSelf)] {
            [weakSelf performCancel];
        });
    }).get();

    return self;
}

#if HAVE(NSPROGRESS_PUBLISHING_SPI)
- (void)_publish
#else
- (void)publish
#endif
{
    if (m_sandboxExtension) {
        BOOL consumedExtension = m_sandboxExtension->consume();
        ASSERT_UNUSED(consumedExtension, consumedExtension);
    }

#if HAVE(NSPROGRESS_PUBLISHING_SPI)
    [super _publish];
#else
    [super publish];
#endif
}

#if HAVE(NSPROGRESS_PUBLISHING_SPI)
- (void)_unpublish
#else
- (void)unpublish
#endif
{
    [self _updateProgressExtendedAttributeOnProgressFile];

#if HAVE(NSPROGRESS_PUBLISHING_SPI)
    [super _unpublish];
#else
    [super unpublish];
#endif

    if (m_sandboxExtension) {
        m_sandboxExtension->revoke();
        m_sandboxExtension = nullptr;
    }
}

- (void)_updateProgressExtendedAttributeOnProgressFile
{
    int64_t total = self.totalUnitCount;
    int64_t completed = self.completedUnitCount;

    float fraction = (total > 0) ? (float)completed / (float)total : -1;
    auto xattrContents = adoptNS([[NSString alloc] initWithFormat:@"%.3f", fraction]);

    setxattr(self.fileURL.fileSystemRepresentation, "com.apple.progress.fractionCompleted", xattrContents.get().UTF8String, [xattrContents.get() lengthOfBytesUsingEncoding:NSUTF8StringEncoding], 0, 0);
}

- (void)dealloc
{
    [m_task.get() removeObserver:self forKeyPath:countOfBytesExpectedToReceiveKeyPath];
    [m_task.get() removeObserver:self forKeyPath:countOfBytesReceivedKeyPath];

    ASSERT(!m_sandboxExtension);

    [super dealloc];
}

- (void)observeValueForKeyPath:(NSString *)keyPath ofObject:(id)object change:(NSDictionary<NSKeyValueChangeKey, id> *)change context:(void *)context
{
    if (context == WKDownloadProgressBytesExpectedToReceiveCountContext) {
        RetainPtr<NSNumber> value = static_cast<NSNumber *>(change[NSKeyValueChangeNewKey]);
        ASSERT([value isKindOfClass:[NSNumber class]]);
        int64_t expectedByteCount = value.get().longLongValue;
        self.totalUnitCount = (expectedByteCount <= 0) ? -1 : expectedByteCount;
    } else if (context == WKDownloadProgressBytesReceivedContext) {
        RetainPtr<NSNumber> value = static_cast<NSNumber *>(change[NSKeyValueChangeNewKey]);
        ASSERT([value isKindOfClass:[NSNumber class]]);
        self.completedUnitCount = value.get().longLongValue;
    } else
        [super observeValueForKeyPath:keyPath ofObject:object change:change context:context];
}

@end
