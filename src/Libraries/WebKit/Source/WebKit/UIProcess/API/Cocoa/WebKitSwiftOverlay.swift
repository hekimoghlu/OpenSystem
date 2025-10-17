/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 26, 2025.
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
#if !os(tvOS) && !os(watchOS)

#if compiler(>=6.0)
internal import WebKit_Private
#endif

#if USE_APPLE_INTERNAL_SDK
@_spi(CTypeConversion) import Network
#endif

@available(iOS 14.0, macOS 10.16, *)
extension WKPDFConfiguration {
    public var rect: CGRect? {
        get { __rect == .null ? nil : __rect }
        set { __rect = newValue == nil ? .null : newValue! }
    }
}

@available(iOS 14.0, macOS 10.16, *)
extension WKWebView {
    @preconcurrency public func callAsyncJavaScript(_ functionBody: String, arguments: [String:Any] = [:], in frame: WKFrameInfo? = nil, in contentWorld: WKContentWorld, completionHandler: (@MainActor (Result<Any, Error>) -> Void)? = nil) {
        let thunk = completionHandler.map { ObjCBlockConversion.boxingNilAsAnyForCompatibility($0) }
        __callAsyncJavaScript(functionBody, arguments: arguments, inFrame: frame, in: contentWorld, completionHandler: thunk)
    }

    @preconcurrency public func createPDF(configuration: WKPDFConfiguration = .init(), completionHandler: @MainActor @escaping (Result<Data, Error>) -> Void) {
        __createPDF(with: configuration, completionHandler: ObjCBlockConversion.exclusive(completionHandler))
    }

    @preconcurrency public func createWebArchiveData(completionHandler: @MainActor @escaping (Result<Data, Error>) -> Void) {
        __createWebArchiveData(completionHandler: ObjCBlockConversion.exclusive(completionHandler))
    }

    @preconcurrency public func evaluateJavaScript(_ javaScript: String, in frame: WKFrameInfo? = nil, in contentWorld: WKContentWorld, completionHandler: (@MainActor (Result<Any, Error>) -> Void)? = nil) {
        let thunk = completionHandler.map { ObjCBlockConversion.boxingNilAsAnyForCompatibility($0) }
        __evaluateJavaScript(javaScript, inFrame: frame, in: contentWorld, completionHandler: thunk)
    }

    @preconcurrency public func find(_ string: String, configuration: WKFindConfiguration = .init(), completionHandler: @MainActor @escaping (WKFindResult) -> Void) {
        __find(string, with: configuration, completionHandler: completionHandler)
    }
}

// Concurrency diagnostics are incorrectly promoted to errors on older public
// versions of Swift. Remove when dropping support for macOS Ventura.
#if (swift(>=5.5) && USE_APPLE_INTERNAL_SDK && NDEBUG) || swift(>=5.10)
@available(iOS 15.0, macOS 12.0, *)
extension WKWebView {
    public func callAsyncJavaScript(_ functionBody: String, arguments: [String:Any] = [:], in frame: WKFrameInfo? = nil, contentWorld: WKContentWorld) async throws -> Any? {
        return try await __callAsyncJavaScript(functionBody, arguments: arguments, inFrame: frame, in: contentWorld)
    }

    public func pdf(configuration: WKPDFConfiguration = .init()) async throws -> Data {
        try await __createPDF(with: configuration)
    }

    public func evaluateJavaScript(_ javaScript: String, in frame: WKFrameInfo? = nil, contentWorld: WKContentWorld) async throws -> Any? {
        try await __evaluateJavaScript(javaScript, inFrame: frame, in: contentWorld)
    }

    public func find(_ string: String, configuration: WKFindConfiguration = .init()) async throws -> WKFindResult {
        await __find(string, with: configuration)
    }
}
#endif

#if compiler(>=6.0)
@available(iOS 18.4, macOS 15.4, visionOS 2.4, *)
@available(watchOS, unavailable)
@available(tvOS, unavailable)
extension WKWebExtension {
    public convenience init(appExtensionBundle: Bundle) async throws {
        // FIXME: <https://webkit.org/b/276194> Make the WebExtension class load data on a background thread.
        try self.init(appExtensionBundle: appExtensionBundle, resourceBaseURL: nil)
    }

    public convenience init(resourceBaseURL: URL) async throws {
        // FIXME: <https://webkit.org/b/276194> Make the WebExtension class load data on a background thread.
        try self.init(appExtensionBundle: nil, resourceBaseURL: resourceBaseURL)
    }
}

@available(iOS 18.4, macOS 15.4, visionOS 2.4, *)
@available(watchOS, unavailable)
@available(tvOS, unavailable)
extension WKWebExtensionController {
    public func didCloseTab(_ closedTab: WKWebExtensionTab, windowIsClosing: Bool = false) {
        __didClose(closedTab, windowIsClosing: windowIsClosing)
    }

    public func didActivateTab(_ activatedTab: any WKWebExtensionTab, previousActiveTab previousTab: (any WKWebExtensionTab)? = nil) {
        __didActivate(activatedTab, previousActiveTab: previousTab)
    }

    public func didMoveTab(_ movedTab: any WKWebExtensionTab, from index: Int, in oldWindow: (any WKWebExtensionWindow)? = nil) {
        __didMove(movedTab, from: index, in: oldWindow)
    }
}

@available(iOS 18.4, macOS 15.4, visionOS 2.4, *)
@available(watchOS, unavailable)
@available(tvOS, unavailable)
extension WKWebExtensionContext {
    public func didCloseTab(_ closedTab: WKWebExtensionTab, windowIsClosing: Bool = false) {
        __didClose(closedTab, windowIsClosing: windowIsClosing)
    }

    public func didActivateTab(_ activatedTab: any WKWebExtensionTab, previousActiveTab previousTab: (any WKWebExtensionTab)? = nil) {
        __didActivate(activatedTab, previousActiveTab: previousTab)
    }

    public func didMoveTab(_ movedTab: any WKWebExtensionTab, from index: Int, in oldWindow: (any WKWebExtensionWindow)? = nil) {
        __didMove(movedTab, from: index, in: oldWindow)
    }
}
#endif

// FIXME: Need to declare ProxyConfiguration SPI in order to build and test
// this with public SDKs (https://bugs.webkit.org/show_bug.cgi?id=280911).
#if USE_APPLE_INTERNAL_SDK
#if canImport(Network, _version: "3623.0.0.0")
@available(iOS 17.0, macOS 14.0, *)
extension WKWebsiteDataStore {
    public var proxyConfigurations: [ProxyConfiguration] {
        get { __proxyConfigurations?.map(ProxyConfiguration.init(_:)) ?? [] }
        set { __proxyConfigurations = newValue.map(\.nw) }
    }
}
#endif
#endif

#endif // !os(tvOS) && !os(watchOS)
