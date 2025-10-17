/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 20, 2023.
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
#import "WKApplicationStateTrackingView.h"
#import "WKBase.h"
#import "WKBrowsingContextController.h"
#import "WKProcessGroup.h"
#import <WebCore/InspectorOverlay.h>
#import <wtf/NakedRef.h>
#import <wtf/RetainPtr.h>
#import <wtf/WeakObjCPtr.h>

@class WKContentView;
@class WKWebView;

namespace API {
class PageConfiguration;
}

namespace WebCore {
class FloatRect;
}

namespace WebKit {
class DrawingAreaProxy;
class RemoteLayerTreeTransaction;
class WebFrameProxy;
class WebPageProxy;
class WebProcessProxy;
class WebProcessPool;
enum class ViewStabilityFlag : uint8_t;
}

@interface WKContentView : WKApplicationStateTrackingView {
@package
    RefPtr<WebKit::WebPageProxy> _page;
    WeakObjCPtr<WKWebView> _webView;
}

ALLOW_DEPRECATED_DECLARATIONS_BEGIN
@property (nonatomic, readonly) WKBrowsingContextController *browsingContextController;
ALLOW_DEPRECATED_DECLARATIONS_END

@property (nonatomic, readonly) WebKit::WebPageProxy* page;
@property (nonatomic, readonly) BOOL isFocusingElement;
@property (nonatomic, getter=isShowingInspectorIndication) BOOL showingInspectorIndication;
@property (nonatomic, readonly, getter=isResigningFirstResponder) BOOL resigningFirstResponder;
@property (nonatomic) BOOL sizeChangedSinceLastVisibleContentRectUpdate;
@property (nonatomic, readonly) UIInterfaceOrientation interfaceOrientation;
@property (nonatomic, readonly) NSUndoManager *undoManagerForWebView;

#if HAVE(SPATIAL_TRACKING_LABEL)
@property (nonatomic, readonly) const String& spatialTrackingLabel;
#endif

- (instancetype)initWithFrame:(CGRect)frame processPool:(NakedRef<WebKit::WebProcessPool>)processPool configuration:(Ref<API::PageConfiguration>&&)configuration webView:(WKWebView *)webView;

- (void)didUpdateVisibleRect:(CGRect)visibleRect
    unobscuredRect:(CGRect)unobscuredRect
    contentInsets:(UIEdgeInsets)contentInsets
    unobscuredRectInScrollViewCoordinates:(CGRect)unobscuredRectInScrollViewCoordinates
    obscuredInsets:(UIEdgeInsets)obscuredInsets
    unobscuredSafeAreaInsets:(UIEdgeInsets)unobscuredSafeAreaInsets
    inputViewBounds:(CGRect)inputViewBounds
    scale:(CGFloat)scale minimumScale:(CGFloat)minimumScale
    viewStability:(OptionSet<WebKit::ViewStabilityFlag>)viewStability
    enclosedInScrollableAncestorView:(BOOL)enclosedInScrollableAncestorView
    sendEvenIfUnchanged:(BOOL)sendEvenIfUnchanged;

- (void)didFinishScrolling;
- (void)didInterruptScrolling;
- (void)didZoomToScale:(CGFloat)scale;
- (void)willStartZoomOrScroll;
- (BOOL)screenIsBeingCaptured;

- (void)_webViewDestroyed;

- (WKWebView *)webView;
- (UIView *)rootContentView;

- (Ref<WebKit::DrawingAreaProxy>)_createDrawingAreaProxy:(WebKit::WebProcessProxy&)webProcessProxy;
- (void)_processDidExit;
#if ENABLE(GPU_PROCESS)
- (void)_gpuProcessDidExit;
#endif
#if ENABLE(MODEL_PROCESS)
- (void)_modelProcessDidExit;
#endif
- (void)_processWillSwap;
- (void)_didRelaunchProcess;

#if HAVE(VISIBILITY_PROPAGATION_VIEW)
- (void)_webProcessDidCreateContextForVisibilityPropagation;
#if ENABLE(GPU_PROCESS)
- (void)_gpuProcessDidCreateContextForVisibilityPropagation;
#endif // ENABLE(GPU_PROCESS)
#if ENABLE(MODEL_PROCESS)
- (void)_modelProcessDidCreateContextForVisibilityPropagation;
#endif // ENABLE(MODEL_PROCESS)
#if USE(EXTENSIONKIT)
- (UIView *)_createVisibilityPropagationView;
#endif
#endif // HAVE(VISIBILITY_PROPAGATION_VIEW)

- (void)_setAcceleratedCompositingRootView:(UIView *)rootView;
- (void)_removeTemporaryDirectoriesWhenDeallocated:(Vector<RetainPtr<NSURL>>&&)urls;

- (void)_showInspectorHighlight:(const WebCore::InspectorOverlay::Highlight&)highlight;
- (void)_hideInspectorHighlight;

- (void)_didCommitLayerTree:(const WebKit::RemoteLayerTreeTransaction&)layerTreeTransaction;
- (void)_layerTreeCommitComplete;

- (void)_setAccessibilityWebProcessToken:(NSData *)data;

- (BOOL)_scrollToRect:(CGRect)targetRect withOrigin:(CGPoint)origin minimumScrollDistance:(CGFloat)minimumScrollDistance;
- (void)_zoomToFocusRect:(CGRect)rectToFocus selectionRect:(CGRect)selectionRect fontSize:(float)fontSize minimumScale:(double)minimumScale maximumScale:(double)maximumScale allowScaling:(BOOL)allowScaling forceScroll:(BOOL)forceScroll;
- (BOOL)_zoomToRect:(CGRect)targetRect withOrigin:(CGPoint)origin fitEntireRect:(BOOL)fitEntireRect minimumScale:(double)minimumScale maximumScale:(double)maximumScale minimumScrollDistance:(CGFloat)minimumScrollDistance;
- (void)_zoomOutWithOrigin:(CGPoint)origin;
- (void)_zoomToInitialScaleWithOrigin:(CGPoint)origin;
- (double)_initialScaleFactor;
- (double)_contentZoomScale;
- (double)_targetContentZoomScaleForRect:(const WebCore::FloatRect&)targetRect currentScale:(double)currentScale fitEntireRect:(BOOL)fitEntireRect minimumScale:(double)minimumScale maximumScale:(double)maximumScale;

@end
