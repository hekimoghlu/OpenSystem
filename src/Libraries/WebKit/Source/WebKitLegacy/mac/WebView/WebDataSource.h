/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 4, 2024.
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
#import <WebKitLegacy/WebDocument.h>

@class NSMutableURLRequest;
@class NSURLConnection;
@class NSURLRequest;
@class NSURLResponse;
@class WebArchive;
@class WebFrame;
@class WebResource;

/*!
    @class WebDataSource
    @discussion A WebDataSource represents the data associated with a web page.
    A datasource has a WebDocumentRepresentation which holds an appropriate
    representation of the data.  WebDataSources manage a hierarchy of WebFrames.
    WebDataSources are typically related to a view by their containing WebFrame.
*/
WEBKIT_CLASS_DEPRECATED_MAC(10_3, 10_14)
@interface WebDataSource : NSObject
{
@package
    void *_private;
}

/*!
    @method initWithRequest:
    @abstract The designated initializer for WebDataSource.
    @param request The request to use in creating a datasource.
    @result Returns an initialized WebDataSource.
*/
- (instancetype)initWithRequest:(NSURLRequest *)request;

/*!
    @property data
    @abstract Returns the raw data associated with the datasource.  Returns nil
    if the datasource hasn't loaded any data.
   @discussion The data will be incomplete until the datasource has completely loaded.
*/
@property (nonatomic, readonly, copy) NSData *data;

/*!
    @property representation
    @abstract The representation associated with this datasource.
    Returns nil if the datasource hasn't created its representation.
    @discussion A representation holds a type specific representation
    of the datasource's data.  The representation class is determined by mapping
    a MIME type to a class.  The representation is created once the MIME type
    of the datasource content has been determined.
*/
@property (nonatomic, readonly, strong) id<WebDocumentRepresentation> representation;

/*!
    @property webFrame
    @abstract The frame that represents this data source.
*/
@property (nonatomic, readonly, strong) WebFrame *webFrame;

/*!
    @property initialRequest
    @abstract A reference to the original request that created the
    datasource.  This request will be unmodified by WebKit. 
*/
@property (nonatomic, readonly, strong) NSURLRequest *initialRequest;

/*!
    @property request
    @abstract The request that was used to create this datasource.
*/
@property (nonatomic, readonly, strong) NSMutableURLRequest *request;

/*!
    @property response
    @abstract The NSURLResponse for the data source.
*/
@property (nonatomic, readonly, strong) NSURLResponse *response;

/*!
    @property textEncodingName
    @abstract Returns either the override encoding, as set on the WebView for this
    dataSource or the encoding from the response.
*/
@property (nonatomic, readonly, copy) NSString *textEncodingName;

/*!
    @property isLoading
    @abstract Returns YES if there are any pending loads.
*/
@property (nonatomic, getter=isLoading, readonly) BOOL loading;

/*!
    @property pageTitle
    @abstract The page title or nil.
*/
@property (nonatomic, readonly, copy) NSString *pageTitle;

/*!
    @property unreachableURL
    @abstract The unreachableURL for which this dataSource is showing alternate content, or nil.
    @discussion This will be non-nil only for dataSources created by calls to the 
    WebFrame method loadAlternateHTMLString:baseURL:forUnreachableURL:.
*/
@property (nonatomic, readonly, strong) NSURL *unreachableURL;

/*!
    @property webArchive
    @abstract A WebArchive representing the data source, its subresources and child frames.
    @description This method constructs a WebArchive using the original downloaded data.
    In the case of HTML, if the current state of the document is preferred, webArchive should be
    called on the DOM document instead.
*/
@property (nonatomic, readonly, strong) WebArchive *webArchive;

/*!
    @property mainResource
    @abstract A WebResource representing the data source.
    @description This method constructs a WebResource using the original downloaded data.
    This method can be used to construct a WebArchive in case the archive returned by
    WebDataSource's webArchive isn't sufficient.
*/
@property (nonatomic, readonly, strong) WebResource *mainResource;

/*!
    @property subresources
    @abstract All the subresources associated with the data source.
    @description The returned array only contains subresources that have fully downloaded.
*/
@property (nonatomic, readonly, copy) NSArray *subresources;

/*!
    method subresourceForURL:
    @abstract Returns a subresource for a given URL.
    @param URL The URL of the subresource.
    @description Returns non-nil if the data source has fully downloaded a subresource with the given URL.
*/
- (WebResource *)subresourceForURL:(NSURL *)URL;

/*!
    @method addSubresource:
    @abstract Adds a subresource to the data source.
    @param subresource The subresource to be added.
    @description addSubresource: adds a subresource to the data source's list of subresources.
    Later, if something causes the data source to load the URL of the subresource, the data source
    will load the data from the subresource instead of from the network. For example, if one wants to add
    an image that is already downloaded to a web page, addSubresource: can be called so that the data source
    uses the downloaded image rather than accessing the network. NOTE: If the data source already has a
    subresource with the same URL, addSubresource: will replace it.
*/
- (void)addSubresource:(WebResource *)subresource;

@end
