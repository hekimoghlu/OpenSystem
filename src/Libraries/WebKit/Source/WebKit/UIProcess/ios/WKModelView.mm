/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 22, 2025.
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
#import "WKModelView.h"

#if ENABLE(ARKIT_INLINE_PREVIEW_IOS)

#import "Logging.h"
#import "RemoteLayerTreeViews.h"
#import "WKModelInteractionGestureRecognizer.h"
#import "WebPageProxy.h"
#import "WebsiteDataStore.h"
#import <WebCore/Model.h>
#import <pal/spi/cocoa/QuartzCoreSPI.h>
#import <pal/spi/ios/SystemPreviewSPI.h>
#import <wtf/Assertions.h>
#import <wtf/FileSystem.h>
#import <wtf/RetainPtr.h>
#import <wtf/SoftLinking.h>
#import <wtf/UUID.h>
#import <wtf/text/MakeString.h>

SOFT_LINK_PRIVATE_FRAMEWORK(AssetViewer);
SOFT_LINK_CLASS(AssetViewer, ASVInlinePreview);

@implementation WKModelView {
    RetainPtr<ASVInlinePreview> _preview;
    RetainPtr<WKModelInteractionGestureRecognizer> _modelInteractionGestureRecognizer;
    String _filePath;
    CGRect _lastBounds;
    Markable<WebCore::PlatformLayerIdentifier> _layerID;
    WeakPtr<WebKit::WebPageProxy> _page;
}

- (ASVInlinePreview *)preview
{
    return _preview.get();
}

- (instancetype)initWithFrame:(CGRect)frame
{
    return nil;
}

- (instancetype)initWithCoder:(NSCoder *)coder
{
    return nil;
}

- (instancetype)initWithModel:(WebCore::Model&)model layerID:(WebCore::PlatformLayerIdentifier)layerID page:(WebKit::WebPageProxy&)page
{
    _lastBounds = CGRectZero;
    self = [super initWithFrame:_lastBounds];
    if (!self)
        return nil;

    _layerID = layerID;
    _page = page;

    [self createFileForModel:model];
    [self updateBounds];

    return self;
}

- (BOOL)createFileForModel:(WebCore::Model&)model
{
    auto pathToDirectory = WebKit::WebsiteDataStore::defaultModelElementCacheDirectory();
    if (pathToDirectory.isEmpty())
        return NO;

    auto directoryExists = FileSystem::fileExists(pathToDirectory);
    if (directoryExists && FileSystem::fileTypeFollowingSymlinks(pathToDirectory) != FileSystem::FileType::Directory) {
        ASSERT_NOT_REACHED();
        return NO;
    }

    if (!directoryExists && !FileSystem::makeAllDirectories(pathToDirectory)) {
        ASSERT_NOT_REACHED();
        return NO;
    }

    String fileName = makeString(WTF::UUID::createVersion4(), ".usdz"_s);
    auto filePath = FileSystem::pathByAppendingComponent(pathToDirectory, fileName);
    auto file = FileSystem::openFile(filePath, FileSystem::FileOpenMode::Truncate);
    if (file <= 0)
        return NO;

    auto byteCount = static_cast<std::size_t>(FileSystem::writeToFile(file, model.data()->span()));
    ASSERT_UNUSED(byteCount, byteCount == model.data()->size());
    FileSystem::closeFile(file);
    _filePath = filePath;

    return YES;
}

- (void)createPreview
{
    if (!_filePath)
        return;

    auto bounds = self.bounds;
    ASSERT(!CGRectEqualToRect(bounds, CGRectZero));

    _preview = adoptNS([allocASVInlinePreviewInstance() initWithFrame:bounds]);
    [self.layer addSublayer:[_preview layer]];

    auto url = adoptNS([[NSURL alloc] initFileURLWithPath:_filePath]);

    [_preview setupRemoteConnectionWithCompletionHandler:^(NSError *contextError) {
        if (contextError) {
            LOG(ModelElement, "Unable to create remote connection, error: %@", [contextError localizedDescription]);
            _page->modelInlinePreviewDidFailToLoad(*_layerID, WebCore::ResourceError { contextError });
            return;
        }

        [_preview preparePreviewOfFileAtURL:url.get() completionHandler:^(NSError *loadError) {
            if (loadError) {
                LOG(ModelElement, "Unable to load file, error: %@", [loadError localizedDescription]);
                _page->modelInlinePreviewDidFailToLoad(*_layerID, WebCore::ResourceError { loadError });
                return;
            }

            LOG(ModelElement, "File loaded successfully.");
            _page->modelInlinePreviewDidLoad(*_layerID);
        }];
    }];

    _modelInteractionGestureRecognizer = adoptNS([[WKModelInteractionGestureRecognizer alloc] init]);
    [self addGestureRecognizer:_modelInteractionGestureRecognizer.get()];
}

- (void)layoutSubviews
{
    [super layoutSubviews];
    [self updateBounds];
}

- (void)updateBounds
{
    auto bounds = self.bounds;
    if (CGRectEqualToRect(_lastBounds, bounds))
        return;

    _lastBounds = bounds;

    if (!_preview) {
        [self createPreview];
        return;
    }

    [_preview updateFrame:bounds completionHandler:^(CAFenceHandle *fenceHandle, NSError *error) {
        if (error) {
            LOG(ModelElement, "Unable to update frame, error: %@", [error localizedDescription]);
            [fenceHandle invalidate];
            return;
        }

        dispatch_async(dispatch_get_main_queue(), ^{
            [self.layer.context addFence:fenceHandle];
            [_preview setFrameWithinFencedTransaction:bounds];
            [fenceHandle invalidate];
        });
    }];
}

- (UIView *)hitTest:(CGPoint)point withEvent:(UIEvent *)event
{
    // The layer of this view is empty and the sublayer is rendered remotely, so the basic implementation
    // of hitTest:withEvent: will return nil due to ignoring empty subviews. So we can simply check whether
    // the hit-testing point is within bounds.
    return [self pointInside:point withEvent:event] ? self : nil;
}

@end

#endif

