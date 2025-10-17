/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 7, 2023.
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
#import "WKImagePreviewViewController.h"

#if PLATFORM(IOS_FAMILY)

#import <UIKitSPI.h>
#import <WebCore/IntSize.h>
#import <_WKElementAction.h>

@implementation WKImagePreviewViewController {
    RetainPtr<CGImageRef> _image;
    RetainPtr<UIImageView> _imageView;
}

- (void)loadView
{
    [super loadView];
    self.view.backgroundColor = [UIColor whiteColor];
    [self.view addSubview:_imageView.get()];
}

- (id)initWithCGImage:(RetainPtr<CGImageRef>)image defaultActions:(RetainPtr<NSArray>)actions elementInfo:(RetainPtr<_WKActivatedElementInfo>)elementInfo
{
    self = [super initWithNibName:nil bundle:nil];
    if (!self)
        return nil;

    _image = image;

    _imageView = adoptNS([[UIImageView alloc] initWithFrame:CGRectZero]);
    _imageView.get().contentMode = UIViewContentModeScaleAspectFill;
    RetainPtr<UIImage> uiImage = adoptNS([[UIImage alloc] initWithCGImage:_image.get()]);
    [_imageView setImage:uiImage.get()];

ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    // FIXME: <rdar://131638772> UIScreen.mainScreen is deprecated.
    CGSize screenSize = [UIScreen mainScreen].bounds.size;
ALLOW_DEPRECATED_DECLARATIONS_END
    CGSize imageSize = _scaleSizeWithinSize(CGSizeMake(CGImageGetWidth(_image.get()), CGImageGetHeight(_image.get())), screenSize);
    [_imageView setFrame:CGRectMake([_imageView frame].origin.x, [_imageView frame].origin.y, imageSize.width, imageSize.height)];
    [self setPreferredContentSize:imageSize];

    _imageActions = actions;
    _activatedElementInfo = elementInfo;

    return self;
}

- (void)viewDidLayoutSubviews
{
    [super viewDidLayoutSubviews];

    [_imageView setFrame:self.view.bounds];
}

static CGSize _scaleSizeWithinSize(CGSize source, CGSize destination)
{
    CGSize size = destination;
    CGFloat sourceAspectRatio = (source.width / source.height);
    CGFloat destinationAspectRatio = (destination.width / destination.height);
    
    if (sourceAspectRatio > destinationAspectRatio) {
        size.width = destination.width;
        size.height = (source.height * (destination.width / source.width));
    } else if (sourceAspectRatio < destinationAspectRatio) {
        size.width = (source.width * (destination.height / source.height));
        size.height = destination.height;
    }
    
    return size;
}

#if HAVE(LINK_PREVIEW)
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
ALLOW_DEPRECATED_IMPLEMENTATIONS_BEGIN
- (NSArray<UIPreviewAction *> *)previewActionItems
{
    NSMutableArray<UIPreviewAction *> *previewActions = [NSMutableArray array];
    for (_WKElementAction *imageAction in _imageActions.get()) {
        UIPreviewAction *previewAction = [UIPreviewAction actionWithTitle:imageAction.title style:UIPreviewActionStyleDefault handler:^(UIPreviewAction *action, UIViewController *previewViewController) {
            [imageAction runActionWithElementInfo:_activatedElementInfo.get()];
        }];
        previewAction.image = [_WKElementAction imageForElementActionType:imageAction.type];

        [previewActions addObject:previewAction];
    }

    return previewActions;
}
ALLOW_DEPRECATED_IMPLEMENTATIONS_END
ALLOW_DEPRECATED_DECLARATIONS_END
#endif

@end

#endif
