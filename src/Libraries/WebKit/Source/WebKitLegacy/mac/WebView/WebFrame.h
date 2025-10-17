/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 11, 2022.
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
#import <Foundation/Foundation.h>
#import <JavaScriptCore/JSBase.h>
#import <WebKitLegacy/WebKitAvailability.h>

@class DOMDocument;
@class DOMHTMLElement;
@class JSContext;
@class NSURLRequest;
@class WebArchive;
@class WebDataSource;
@class WebFramePrivate;
@class WebFrameView;
@class WebScriptObject;
@class WebView;

/*!
    @class WebFrame
    @discussion Every web page is represented by at least one WebFrame.  A WebFrame
    has a WebFrameView and a WebDataSource.
*/
WEBKIT_CLASS_DEPRECATED_MAC(10_3, 10_14)
@interface WebFrame : NSObject
{
@package
    WebFramePrivate *_private;
}

/*!
    @method initWithName:webFrameView:webView:
    @abstract The designated initializer of WebFrame.
    @discussion WebFrames are normally created for you by the WebView.  You should 
    not need to invoke this method directly.
    @param name The name of the frame.
    @param view The WebFrameView for the frame.
    @param webView The WebView that manages the frame.
    @result Returns an initialized WebFrame.
*/
- (instancetype)initWithName:(NSString *)name webFrameView:(WebFrameView *)view webView:(WebView *)webView;

/*!
    @property name
    @abstract The frame name.
*/
@property (nonatomic, readonly, copy) NSString *name;

/*!
    @property webView
    @abstract The WebView for the document that includes this frame.
*/
@property (nonatomic, readonly, strong) WebView *webView;

/*!
    @property frameView
    @abstract The WebFrameView for this frame.
*/
@property (nonatomic, readonly, strong) WebFrameView *frameView;

/*!
    @property DOMDocument
    @abstract The DOM document of the frame.
    @description Returns nil if the frame does not contain a DOM document such as a standalone image.
*/
@property (nonatomic, readonly, strong) DOMDocument *DOMDocument;

/*!
    @property frameElement
    @abstract The frame element of the frame.
    @description The class of the result is either DOMHTMLFrameElement, DOMHTMLIFrameElement or DOMHTMLObjectElement.
    Returns nil if the frame is the main frame since there is no frame element for the frame in this case.
*/
@property (nonatomic, readonly, strong) DOMHTMLElement *frameElement;

/*!
    @method loadRequest:
    @param request The web request to load.
*/
- (void)loadRequest:(NSURLRequest *)request;

/*!
    @method loadData:MIMEType:textEncodingName:baseURL:
    @param data The data to use for the main page of the document.
    @param MIMEType The MIME type of the data.
    @param encodingName The encoding of the data.
    @param URL The base URL to apply to relative URLs within the document.
*/
- (void)loadData:(NSData *)data MIMEType:(NSString *)MIMEType textEncodingName:(NSString *)encodingName baseURL:(NSURL *)URL;

/*!
    @method loadHTMLString:baseURL:
    @param string The string to use for the main page of the document.
    @param URL The base URL to apply to relative URLs within the document.
*/
- (void)loadHTMLString:(NSString *)string baseURL:(NSURL *)URL;

/*!
    @method loadAlternateHTMLString:baseURL:forUnreachableURL:
    @abstract Loads a page to display as a substitute for a URL that could not be reached.
    @discussion This allows clients to display page-loading errors in the webview itself.
    This is typically called while processing the WebFrameLoadDelegate method
    -webView:didFailProvisionalLoadWithError:forFrame: or one of the WebPolicyDelegate methods
    -webView:decidePolicyForMIMEType:request:frame:decisionListener: or
    -webView:unableToImplementPolicyWithError:frame:. If it is called from within one of those
    three delegate methods then the back/forward list will be maintained appropriately.
    @param string The string to use for the main page of the document.
    @param baseURL The baseURL to apply to relative URLs within the document.
    @param unreachableURL The URL for which this page will serve as alternate content.
*/
- (void)loadAlternateHTMLString:(NSString *)string baseURL:(NSURL *)baseURL forUnreachableURL:(NSURL *)unreachableURL;

/*!
    @method loadArchive:
    @abstract Causes WebFrame to load a WebArchive.
    @param archive The archive to be loaded.
*/
- (void)loadArchive:(WebArchive *)archive;

/*!
    @property dataSource
    @abstract The datasource for this frame.
    @discussion Returns the committed data source.  Will return nil if the
    provisional data source hasn't yet been loaded.
*/
@property (nonatomic, readonly, strong) WebDataSource *dataSource;

/*!
    @property provisionalDataSource
    @abstract The provisional datasource of this frame.
    @discussion Will return the provisional data source.  The provisional data source will
    be nil if no data source has been set on the frame, or the data source
    has successfully transitioned to the committed data source.
*/
@property (nonatomic, readonly, strong) WebDataSource *provisionalDataSource;

/*!
    @method stopLoading
    @discussion Stop any pending loads on the frame's data source,
    and its children.
*/
- (void)stopLoading;

/*!
    @method reload
    @discussion Performs HTTP/1.1 end-to-end revalidation using cache-validating conditionals if possible.
*/
- (void)reload;

/*!
    @method reloadFromOrigin
    @discussion Performs HTTP/1.1 end-to-end reload.
*/
- (void)reloadFromOrigin;

/*!
    @method findFrameNamed:
    @discussion This method returns a frame with the given name. findFrameNamed returns self 
    for _self and _current, the parent frame for _parent and the main frame for _top. 
    findFrameNamed returns self for _parent and _top if the receiver is the mainFrame.
    findFrameNamed first searches from the current frame to all descending frames then the
    rest of the frames in the WebView. If still not found, findFrameNamed searches the
    frames of the other WebViews.
    @param name The name of the frame to find.
    @result The frame matching the provided name. nil if the frame is not found.
*/
- (WebFrame *)findFrameNamed:(NSString *)name;

/*!
    @property parentFrame
    @abstract The frame containing this frame, or nil if this is a top level frame.
*/
@property (nonatomic, readonly, strong) WebFrame *parentFrame;

/*!
    @property childFrames
    @abstract An array of WebFrame.
    @discussion The frames in the array are associated with a frame set or iframe.
*/
@property (nonatomic, readonly, copy) NSArray *childFrames;

/*!
    @property windowObject
    @abstract The WebScriptObject representing the frame's JavaScript window object.
*/
@property (nonatomic, readonly, strong) WebScriptObject *windowObject;

/*!
    @property globalContext
    @abstract The frame's global JavaScript execution context.
    @discussion Use this method to bridge between the WebKit and JavaScriptCore APIs.
*/
@property (nonatomic, readonly) JSGlobalContextRef globalContext;

#if JSC_OBJC_API_ENABLED
/*!
    @property javaScriptContext
    @abstract The frame's global JavaScript execution context.
    @discussion Use this method to bridge between the WebKit and Objective-C JavaScriptCore API.
*/
@property (nonatomic, readonly, strong) JSContext *javaScriptContext;
#endif // JSC_OBJC_API_ENABLED

@end
