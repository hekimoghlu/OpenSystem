/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 22, 2024.
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
#pragma once

#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/ThreadSafeRefCounted.h>

#if PLATFORM(COCOA)
#include "WKFoundation.h"
#ifdef __OBJC__
#include "WKObject.h"
#include <wtf/RetainPtr.h>
#endif
#include <wtf/HashMap.h>
#endif

#define DELEGATE_REF_COUNTING_TO_COCOA PLATFORM(COCOA)

namespace API {

class Object
#if !DELEGATE_REF_COUNTING_TO_COCOA
    : public ThreadSafeRefCounted<Object>
#endif
{
    WTF_MAKE_NONCOPYABLE(Object);
public:
    enum class Type {
        // Base types
        Null = 0,
        Array,
        AuthenticationChallenge,
        AuthenticationDecisionListener,
        CaptionUserPreferencesTestingModeToken,
        CertificateInfo,
        ContextMenuItem,
        Credential,
        Data,
        Dictionary,
        Error,
        FrameHandle,
        Image,
        PageHandle,
        ProtectionSpace,
        RenderLayer,
        RenderObject,
        ResourceLoadInfo,
        SecurityOrigin,
        SessionState,
        SerializedScriptValue,
        String,
        TargetedElementInfo,
        TargetedElementRequest,
        URL,
        URLRequest,
        URLResponse,
        UserContentURLPattern,
        UserScript,
        UserStyleSheet,
        WebArchive,
        WebArchiveResource,

        // Base numeric types
        Boolean,
        Double,
        UInt64,
        Int64,
        
        // Geometry types
        Point,
        Size,
        Rect,
        
        // UIProcess types
        ApplicationCacheManager,
#if ENABLE(APPLICATION_MANIFEST)
        ApplicationManifest,
#endif
        Attachment,
        AutomationSession,
        BackForwardList,
        BackForwardListItem,
        CacheManager,
        ColorPickerResultListener,
        ContentRuleList,
        ContentRuleListAction,
        ContentRuleListStore,
        ContentWorld,
#if PLATFORM(IOS_FAMILY)
        ContextMenuElementInfo,
#endif
#if PLATFORM(MAC)
        ContextMenuElementInfoMac,
#endif
        ContextMenuListener,
        CustomHeaderFields,
        DataTask,
        DebuggableInfo,
        Download,
        Feature,
        FormSubmissionListener,
        Frame,
        FrameInfo,
        FramePolicyListener,
        FrameTreeNode,
        FullScreenManager,
        GeolocationManager,
        GeolocationPermissionRequest,
        HTTPCookieStore,
        HitTestResult,
        GeolocationPosition,
        GrammarDetail,
        IconDatabase,
        Inspector,
        InspectorConfiguration,
#if ENABLE(INSPECTOR_EXTENSIONS)
        InspectorExtension,
#endif
        KeyValueStorageManager,
        MediaCacheManager,
        MessageListener,
        Navigation,
        NavigationAction,
        NavigationData,
        NavigationResponse,
        Notification,
        NotificationManager,
        NotificationPermissionRequest,
        OpenPanelParameters,
        OpenPanelResultListener,
        OriginDataManager,
        Page,
        PageConfiguration,
        PageGroup,
        ProcessPool,
        ProcessPoolConfiguration,
        PluginSiteDataManager,
        Preferences,
        RequestStorageAccessConfirmResultListener,
        ResourceLoadStatisticsStore,
        ResourceLoadStatisticsFirstParty,
        ResourceLoadStatisticsThirdParty,
        RunBeforeUnloadConfirmPanelResultListener,
        RunJavaScriptAlertResultListener,
        RunJavaScriptConfirmResultListener,
        RunJavaScriptPromptResultListener,
        SpeechRecognitionPermissionCallback,
        TextChecker,
        TextRun,
        URLSchemeTask,
        UserContentController,
        UserInitiatedAction,
        UserMediaPermissionCheck,
        UserMediaPermissionRequest,
        ViewportAttributes,
        VisitedLinkStore,
#if ENABLE(WK_WEB_EXTENSIONS)
        WebExtension,
        WebExtensionAction,
        WebExtensionCommand,
        WebExtensionContext,
        WebExtensionController,
        WebExtensionControllerConfiguration,
        WebExtensionDataRecord,
        WebExtensionMatchPattern,
        WebExtensionMessagePort,
#if ENABLE(WK_WEB_EXTENSIONS_SIDEBAR)
        WebExtensionSidebar,
#endif
#endif
        WebResourceLoadStatisticsManager,
        WebPushDaemonConnection,
        WebPushMessage,
        WebPushSubscriptionData,
        WebsiteDataRecord,
        WebsiteDataStore,
        WebsiteDataStoreConfiguration,
        WebsitePolicies,
        WindowFeatures,

#if ENABLE(WEB_AUTHN)
        WebAuthenticationAssertionResponse,
        WebAuthenticationPanel,
#endif

        MediaKeySystemPermissionCallback,
        QueryPermissionResultCallback,

        // Bundle types
        Bundle,
        BundleBackForwardList,
        BundleBackForwardListItem,
        BundleCSSStyleDeclarationHandle,
        BundleDOMWindowExtension,
        BundleFrame,
        BundleHitTestResult,
        BundleNodeHandle,
        BundlePage,
        BundlePageBanner,
        BundlePageOverlay,
        BundleRangeHandle,
        BundleScriptWorld,

        // Platform specific
        EditCommandProxy,
        View,
#if USE(SOUP)
        SoupRequestManager,
        SoupCustomProtocolRequestManager,
#endif
    };

    virtual ~Object() = default;

    virtual Type type() const = 0;

#if DELEGATE_REF_COUNTING_TO_COCOA
#ifdef __OBJC__
    template<typename T, typename... Args>
    static void constructInWrapper(id <WKObject> wrapper, Args&&... args)
    {
        Object& object = wrapper._apiObject;

        apiObjectsUnderConstruction().add(&object, (__bridge CFTypeRef)wrapper);

        new (&object) T(std::forward<Args>(args)...);
    }

    id <WKObject> wrapper() const { return (__bridge id <WKObject>)m_wrapper; }
#endif

    void ref() const;
    void deref() const;
#endif // DELEGATE_REF_COUNTING_TO_COCOA

    static void* wrap(API::Object*);
    static API::Object* unwrap(void*);

#if PLATFORM(COCOA) && defined(__OBJC__)
    RetainPtr<NSObject<NSSecureCoding>> toNSObject();
    static RefPtr<API::Object> fromNSObject(NSObject<NSSecureCoding> *);

    static API::Object& fromWKObjectExtraSpace(id <WKObject>);
#endif

protected:
    Object();

#if DELEGATE_REF_COUNTING_TO_COCOA
    static void* newObject(size_t, Type);

private:
    static HashMap<Object*, CFTypeRef>& apiObjectsUnderConstruction();

    // Derived classes must override operator new and call newObject().
    void* operator new(size_t) = delete;

    CFTypeRef m_wrapper;
#endif // DELEGATE_REF_COUNTING_TO_COCOA
};

template <Object::Type ArgumentType>
class ObjectImpl : public Object {
public:
    static const Type APIType = ArgumentType;

protected:
    friend class Object;

    ObjectImpl() = default;

    Type type() const override { return APIType; }

#if DELEGATE_REF_COUNTING_TO_COCOA
    void* operator new(size_t size) { return newObject(size, APIType); }
    void* operator new(size_t, void* value) { return value; }
#endif
};

#if !DELEGATE_REF_COUNTING_TO_COCOA
inline void* Object::wrap(API::Object* object)
{
    return static_cast<void*>(object);
}

inline API::Object* Object::unwrap(void* object)
{
    return static_cast<API::Object*>(object);
}
#endif

} // namespace API

#undef DELEGATE_REF_COUNTING_TO_COCOA

#define SPECIALIZE_TYPE_TRAITS_API_OBJECT(ClassName) \
SPECIALIZE_TYPE_TRAITS_BEGIN(API::ClassName) \
static bool isType(const API::Object& object) { return object.type() == API::Object::Type::ClassName; } \
SPECIALIZE_TYPE_TRAITS_END()
