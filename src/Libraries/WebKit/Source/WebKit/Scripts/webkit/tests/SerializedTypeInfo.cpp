/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 27, 2023.
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
#include "config.h"
#include "SerializedTypeInfo.h"

#include "CommonHeader.h"
#if ENABLE(TEST_FEATURE)
#include "CommonHeader.h"
#endif
#include "CustomEncoded.h"
#if ENABLE(TEST_FEATURE)
#include "FirstMemberType.h"
#endif
#include "FooWrapper.h"
#include "FormDataReference.h"
#include "GeneratedWebKitSecureCoding.h"
#include "HeaderWithoutCondition"
#include "LayerProperties.h"
#include "PlatformClass.h"
#include "RValueWithFunctionCalls.h"
#include "RemoteVideoFrameIdentifier.h"
#if ENABLE(TEST_FEATURE)
#include "SecondMemberType.h"
#endif
#if ENABLE(TEST_FEATURE)
#include "StructHeader.h"
#endif
#include "TemplateTest.h"
#include <Namespace/EmptyConstructorStruct.h>
#include <Namespace/EmptyConstructorWithIf.h>
#if !(ENABLE(OUTER_CONDITION))
#include <Namespace/OtherOuterClass.h>
#endif
#if ENABLE(OUTER_CONDITION)
#include <Namespace/OuterClass.h>
#endif
#include <Namespace/ReturnRefClass.h>
#if USE(APPKIT)
#include <WebCore/AppKitControlSystemImage.h>
#endif
#include <WebCore/FloatBoxExtent.h>
#include <WebCore/InheritanceGrandchild.h>
#include <WebCore/InheritsFrom.h>
#include <WebCore/MoveOnlyBaseClass.h>
#include <WebCore/MoveOnlyDerivedClass.h>
#if USE(APPKIT)
#include <WebCore/ScrollbarTrackCornerSystemImageMac.h>
#endif
#include <WebCore/ScrollingStateFrameHostingNode.h>
#include <WebCore/ScrollingStateFrameHostingNodeWithStuffAfterTuple.h>
#include <WebCore/TimingFunction.h>
#if USE(AVFOUNDATION)
#include <pal/cocoa/AVFoundationSoftLink.h>
#endif
#if ENABLE(DATA_DETECTION)
#include <pal/cocoa/DataDetectorsCoreSoftLink.h>
#endif
#include <wtf/CreateUsingClass.h>
#include <wtf/Seconds.h>

static_assert(std::is_same_v<WebCore::SharedStringHash,
    uint32_t
>);
static_assert(std::is_same_v<WebCore::UsingWithSemicolon,
    uint32_t
>);
#if OS(WINDOWS)
static_assert(std::is_same_v<WTF::ProcessID,
    int
>);
#endif
#if !(OS(WINDOWS))
static_assert(std::is_same_v<WTF::ProcessID,
    pid_t
>);
#endif
static_assert(std::is_same_v<WebCore::ConditionalVariant,
    std::variant<
        int,
#if USE(CHAR)
        char,
#endif
        double
    >
>);
static_assert(std::is_same_v<WebCore::NonConditionalVariant,
    std::variant<int, double>
>);

#if ENABLE(IPC_TESTING_API)

namespace WebKit {

template<typename E> uint64_t enumValueForIPCTestAPI(E e)
{
    return static_cast<std::make_unsigned_t<std::underlying_type_t<E>>>(e);
}

Vector<SerializedTypeInfo> allSerializedTypes()
{
    return {
#if ENABLE(TEST_FEATURE)
        { "Namespace::Subnamespace::StructName"_s, {
            {
                "FirstMemberType"_s,
                "firstMemberName"_s
            },
#if ENABLE(SECOND_MEMBER)
            {
                "SecondMemberType"_s,
                "secondMemberName"_s
            },
#endif
            {
                "RetainPtr<CFTypeRef>"_s,
                "nullableTestMember"_s
            },
        } },
#endif // ENABLE(TEST_FEATURE)
        { "Namespace::OtherClass"_s, {
            {
                "int"_s,
                "a"_s
            },
            {
                "bool"_s,
                "b"_s
            },
            {
                "RetainPtr<NSArray>"_s,
                "dataDetectorResults"_s
            },
        } },
        { "Namespace::ClassWithMemberPrecondition"_s, {
            {
                "RetainPtr<PKPaymentMethod>"_s,
                "m_pkPaymentMethod"_s
            },
        } },
        { "Namespace::ReturnRefClass"_s, {
            {
                "double"_s,
                "functionCall().member1"_s
            },
            {
                "double"_s,
                "functionCall().member2"_s
            },
            {
                "std::unique_ptr<int>"_s,
                "uniqueMember"_s
            },
        } },
        { "Namespace::EmptyConstructorStruct"_s, {
            {
                "int"_s,
                "m_int"_s
            },
            {
                "double"_s,
                "m_double"_s
            },
        } },
        { "Namespace::EmptyConstructorWithIf"_s, {
#if CONDITION_AROUND_M_TYPE_AND_M_VALUE
            {
                "MemberType"_s,
                "m_type"_s
            },
#endif
#if CONDITION_AROUND_M_TYPE_AND_M_VALUE
            {
                "OtherMemberType"_s,
                "m_value"_s
            },
#endif
        } },
        { "WithoutNamespace"_s, {
            {
                "int"_s,
                "a"_s
            },
        } },
        { "WithoutNamespaceWithAttributes"_s, {
            {
                "int"_s,
                "a"_s
            },
        } },
        { "WebCore::InheritsFrom"_s, {
            {
                "int"_s,
                "a"_s
            },
            {
                "float"_s,
                "b"_s
            },
        } },
        { "WebCore::InheritanceGrandchild"_s, {
            {
                "int"_s,
                "a"_s
            },
            {
                "float"_s,
                "b"_s
            },
            {
                "double"_s,
                "c"_s
            },
        } },
        { "Seconds"_s, {
            {
                "double"_s,
                "value()"_s
            },
        } },
        { "CreateUsingClass"_s, {
            {
                "double"_s,
                "value"_s
            },
        } },
        { "WebCore::FloatBoxExtent"_s, {
            {
                "float"_s,
                "top()"_s
            },
            {
                "float"_s,
                "right()"_s
            },
            {
                "float"_s,
                "bottom()"_s
            },
            {
                "float"_s,
                "left()"_s
            },
        } },
        { "SoftLinkedMember"_s, {
            {
                "RetainPtr<DDActionContext>"_s,
                "firstMember"_s
            },
            {
                "RetainPtr<DDActionContext>"_s,
                "secondMember"_s
            },
        } },
        { "WebCore::TimingFunction"_s, {
            { "std::variant<"
                "WebCore::LinearTimingFunction"
                ", WebCore::CubicBezierTimingFunction"
#if CONDITION
                ", WebCore::StepsTimingFunction"
#endif
                ", WebCore::SpringTimingFunction"
            ">"_s, "subclasses"_s }
        } },
#if ENABLE(TEST_FEATURE)
        { "Namespace::ConditionalCommonClass"_s, {
            {
                "int"_s,
                "value"_s
            },
        } },
#endif // ENABLE(TEST_FEATURE)
        { "Namespace::CommonClass"_s, {
            {
                "int"_s,
                "value"_s
            },
        } },
        { "Namespace::AnotherCommonClass"_s, {
            {
                "int"_s,
                "value"_s
            },
        } },
        { "WebCore::MoveOnlyBaseClass"_s, {
            { "std::variant<"
                "WebCore::MoveOnlyDerivedClass"
            ">"_s, "subclasses"_s }
        } },
        { "WebCore::MoveOnlyDerivedClass"_s, {
            {
                "int"_s,
                "firstMember"_s
            },
            {
                "int"_s,
                "secondMember"_s
            },
        } },
        { "WebKit::PlatformClass"_s, {
            {
                "int"_s,
                "value"_s
            },
        } },
        { "WebKit::CustomEncoded"_s, {
            {
                "int"_s,
                "value"_s
            },
        } },
        { "WebKit::LayerProperties"_s, {
            {
                "OptionalTuple<"
                    "String"
#if ENABLE(FEATURE)
                    ", std::unique_ptr<WebCore::TransformationMatrix>"
#endif
                    ", bool"
                ">"_s,
                "OptionalTuple<"
                    "name"
#if ENABLE(FEATURE)
                    ", featureEnabledMember"
#endif
                    ", bitFieldMember"
                ">"_s
            },
        } },
        { "WebKit::TemplateTest"_s, {
            {
                "bool"_s,
                "value"_s
            },
        } },
        { "WebCore::ScrollingStateFrameHostingNode"_s, {
            {
                "WebCore::ScrollingNodeID"_s,
                "scrollingNodeID()"_s
            },
            {
                "Vector<Ref<WebCore::ScrollingStateNode>>"_s,
                "children()"_s
            },
            {
                "OptionalTuple<"
                    "std::optional<WebCore::PlatformLayerIdentifier>"
                ">"_s,
                "OptionalTuple<"
                    "layer().layerIDForEncoding()"
                ">"_s
            },
        } },
        { "WebCore::ScrollingStateFrameHostingNodeWithStuffAfterTuple"_s, {
            {
                "WebCore::ScrollingNodeID"_s,
                "scrollingNodeID()"_s
            },
            {
                "Vector<Ref<WebCore::ScrollingStateNode>>"_s,
                "children()"_s
            },
            {
                "OptionalTuple<"
                    "std::optional<WebCore::PlatformLayerIdentifier>"
                    ", bool"
                ">"_s,
                "OptionalTuple<"
                    "layer().layerIDForEncoding()"
                    ", otherMember"
                ">"_s
            },
            {
                "int"_s,
                "memberAfterTuple"_s
            },
        } },
        { "RequestEncodedWithBody"_s, {
            {
                "WebCore::ResourceRequest"_s,
                "request"_s
            },
            {
                "IPC::FormDataReference"_s,
                "requestBody"_s
            },
        } },
        { "RequestEncodedWithBodyRValue"_s, {
            {
                "WebCore::ResourceRequest"_s,
                "request"_s
            },
            {
                "IPC::FormDataReference"_s,
                "requestBody"_s
            },
        } },
#if USE(AVFOUNDATION)
        { "WebKit::CoreIPCAVOutputContext"_s, {
            { "RetainPtr<NSString>"_s , "AVOutputContextSerializationKeyContextID"_s },
            { "RetainPtr<NSString>"_s , "AVOutputContextSerializationKeyContextType"_s },
        } },
        { "AVOutputContext"_s, {
            { "WebKit::CoreIPCAVOutputContext"_s, "wrapper"_s }
        } },
#endif // USE(AVFOUNDATION)
        { "WebKit::CoreIPCNSSomeFoundationType"_s, {
            { "RetainPtr<NSString>"_s , "StringKey"_s },
            { "RetainPtr<NSNumber>"_s , "NumberKey"_s },
            { "RetainPtr<NSNumber>"_s , "OptionalNumberKey"_s },
            { "RetainPtr<NSArray>"_s , "ArrayKey"_s },
            { "RetainPtr<NSArray>"_s , "OptionalArrayKey"_s },
            { "RetainPtr<NSDictionary>"_s , "DictionaryKey"_s },
            { "RetainPtr<NSDictionary>"_s , "OptionalDictionaryKey"_s },
        } },
        { "NSSomeFoundationType"_s, {
            { "WebKit::CoreIPCNSSomeFoundationType"_s, "wrapper"_s }
        } },
        { "WebKit::CoreIPCclass NSSomeOtherFoundationType"_s, {
            { "RetainPtr<NSDictionary>"_s , "DictionaryKey"_s },
        } },
        { "class NSSomeOtherFoundationType"_s, {
            { "WebKit::CoreIPCclass NSSomeOtherFoundationType"_s, "wrapper"_s }
        } },
#if ENABLE(DATA_DETECTION)
        { "WebKit::CoreIPCDDScannerResult"_s, {
            { "RetainPtr<NSString>"_s , "StringKey"_s },
            { "RetainPtr<NSNumber>"_s , "NumberKey"_s },
            { "RetainPtr<NSNumber>"_s , "OptionalNumberKey"_s },
            { "Vector<RetainPtr<DDScannerResult>>"_s , "ArrayKey"_s },
            { "std::optional<Vector<RetainPtr<DDScannerResult>>>"_s , "OptionalArrayKey"_s },
            { "Vector<std::pair<String, RetainPtr<Number>>>"_s , "DictionaryKey"_s },
            { "std::optional<Vector<std::pair<String, RetainPtr<DDScannerResult>>>>"_s , "OptionalDictionaryKey"_s },
            { "Vector<RetainPtr<NSData>>"_s , "DataArrayKey"_s },
            { "Vector<RetainPtr<SecTrustRef>>"_s , "SecTrustArrayKey"_s },
        } },
        { "DDScannerResult"_s, {
            { "WebKit::CoreIPCDDScannerResult"_s, "wrapper"_s }
        } },
#endif // ENABLE(DATA_DETECTION)
        { "CFFooRef"_s, {
            { "WebKit::FooWrapper"_s, "wrapper"_s }
        } },
#if USE(CFBAR)
        { "CFBarRef"_s, {
            { "WebKit::BarWrapper"_s, "wrapper"_s }
        } },
#endif // USE(CFBAR)
#if USE(CFSTRING)
        { "CFStringRef"_s, {
            { "String"_s, "wrapper"_s }
        } },
#endif // USE(CFSTRING)
#if USE(SKIA)
        { "SkFooBar"_s, {
            {
                "int"_s,
                "foo()"_s
            },
            {
                "double"_s,
                "bar()"_s
            },
        } },
#endif // USE(SKIA)
        { "WebKit::RValueWithFunctionCalls"_s, {
            {
                "SandboxExtensionHandle"_s,
                "callFunction()"_s
            },
        } },
        { "WebKit::RemoteVideoFrameReference"_s, {
            {
                "WebKit::RemoteVideoFrameIdentifier"_s,
                "identifier()"_s
            },
            {
                "uint64_t"_s,
                "version()"_s
            },
        } },
        { "WebKit::RemoteVideoFrameWriteReference"_s, {
            {
                "IPC::ObjectIdentifierReference<WebKit::RemoteVideoFrameIdentifier>"_s,
                "reference()"_s
            },
            {
                "uint64_t"_s,
                "pendingReads()"_s
            },
        } },
#if ENABLE(OUTER_CONDITION)
        { "Namespace::OuterClass"_s, {
            {
                "int"_s,
                "outerValue"_s
            },
        } },
#endif // ENABLE(OUTER_CONDITION)
#if !(ENABLE(OUTER_CONDITION))
        { "Namespace::OtherOuterClass"_s, {
            {
                "int"_s,
                "outerValue"_s
            },
        } },
#endif // !(ENABLE(OUTER_CONDITION))
#if USE(APPKIT)
        { "WebCore::AppKitControlSystemImage"_s, {
            {
                "WebCore::Color"_s,
                "m_tintColor"_s
            },
            {
                "bool"_s,
                "m_useDarkAppearance"_s
            },
        } },
#endif // USE(APPKIT)
        { "WebCore::RectEdges<bool>"_s, {
            {
                "bool"_s,
                "top()"_s
            },
            {
                "bool"_s,
                "right()"_s
            },
            {
                "bool"_s,
                "bottom()"_s
            },
            {
                "bool"_s,
                "left()"_s
            },
        } },
#if USE(PASSKIT)
        { "PKPaymentMethod"_s, {
            { "WebKit::CoreIPCPKPaymentMethod"_s, "wrapper"_s }
        } },
#endif // USE(PASSKIT)
        { "NSNull"_s, {
            { "WebKit::CoreIPCNull"_s, "wrapper"_s }
        } },
        { "WebCore::SharedStringHash"_s, {
        {
            "uint32_t"_s
            , "alias"_s }
        } },
        { "WebCore::UsingWithSemicolon"_s, {
        {
            "uint32_t"_s
            , "alias"_s }
        } },
#if OS(WINDOWS)
        { "WTF::ProcessID"_s, {
        {
            "int"_s
            , "alias"_s }
        } },
#endif
#if !(OS(WINDOWS))
        { "WTF::ProcessID"_s, {
        {
            "pid_t"_s
            , "alias"_s }
        } },
#endif
        { "WebCore::ConditionalVariant"_s, {
        {
            "std::variant<"
            "int, "
#if USE(CHAR)
            "char, "
#endif
            "double"
            ">"_s
            , "alias"_s }
        } },
        { "WebCore::NonConditionalVariant"_s, {
        {
            "std::variant<int, double>"_s
            , "alias"_s }
        } },
    };
}

Vector<SerializedEnumInfo> allSerializedEnums()
{
    return {
#if ENABLE(BOOL_ENUM)
        { "EnumNamespace::BoolEnumType"_s, sizeof(EnumNamespace::BoolEnumType), false, {
            0, 1
        } },
#endif
        { "EnumWithoutNamespace"_s, sizeof(EnumWithoutNamespace), false, {
            enumValueForIPCTestAPI(EnumWithoutNamespace::Value1),
            enumValueForIPCTestAPI(EnumWithoutNamespace::Value2),
            enumValueForIPCTestAPI(EnumWithoutNamespace::Value3),
        } },
#if ENABLE(UINT16_ENUM)
        { "EnumNamespace::EnumType"_s, sizeof(EnumNamespace::EnumType), false, {
            enumValueForIPCTestAPI(EnumNamespace::EnumType::FirstValue),
#if ENABLE(ENUM_VALUE_CONDITION)
            enumValueForIPCTestAPI(EnumNamespace::EnumType::SecondValue),
#endif
        } },
#endif
        { "EnumNamespace2::OptionSetEnumType"_s, sizeof(EnumNamespace2::OptionSetEnumType), true, {
            enumValueForIPCTestAPI(EnumNamespace2::OptionSetEnumType::OptionSetFirstValue),
#if ENABLE(OPTION_SET_SECOND_VALUE)
            enumValueForIPCTestAPI(EnumNamespace2::OptionSetEnumType::OptionSetSecondValue),
#endif
#if !(ENABLE(OPTION_SET_SECOND_VALUE))
            enumValueForIPCTestAPI(EnumNamespace2::OptionSetEnumType::OptionSetSecondValueElse),
#endif
            enumValueForIPCTestAPI(EnumNamespace2::OptionSetEnumType::OptionSetThirdValue),
        } },
        { "OptionSetEnumFirstCondition"_s, sizeof(OptionSetEnumFirstCondition), true, {
#if ENABLE(OPTION_SET_FIRST_VALUE)
            enumValueForIPCTestAPI(OptionSetEnumFirstCondition::OptionSetFirstValue),
#endif
            enumValueForIPCTestAPI(OptionSetEnumFirstCondition::OptionSetSecondValue),
            enumValueForIPCTestAPI(OptionSetEnumFirstCondition::OptionSetThirdValue),
        } },
        { "OptionSetEnumLastCondition"_s, sizeof(OptionSetEnumLastCondition), true, {
            enumValueForIPCTestAPI(OptionSetEnumLastCondition::OptionSetFirstValue),
            enumValueForIPCTestAPI(OptionSetEnumLastCondition::OptionSetSecondValue),
#if ENABLE(OPTION_SET_THIRD_VALUE)
            enumValueForIPCTestAPI(OptionSetEnumLastCondition::OptionSetThirdValue),
#endif
        } },
        { "OptionSetEnumAllCondition"_s, sizeof(OptionSetEnumAllCondition), true, {
#if ENABLE(OPTION_SET_FIRST_VALUE)
            enumValueForIPCTestAPI(OptionSetEnumAllCondition::OptionSetFirstValue),
#endif
#if ENABLE(OPTION_SET_SECOND_VALUE)
            enumValueForIPCTestAPI(OptionSetEnumAllCondition::OptionSetSecondValue),
#endif
#if ENABLE(OPTION_SET_THIRD_VALUE)
            enumValueForIPCTestAPI(OptionSetEnumAllCondition::OptionSetThirdValue),
#endif
        } },
#if (ENABLE(OUTER_CONDITION)) && (ENABLE(INNER_CONDITION))
        { "EnumNamespace::InnerEnumType"_s, sizeof(EnumNamespace::InnerEnumType), false, {
            enumValueForIPCTestAPI(EnumNamespace::InnerEnumType::InnerValue),
#if ENABLE(INNER_INNER_CONDITION)
            enumValueForIPCTestAPI(EnumNamespace::InnerEnumType::InnerInnerValue),
#endif
#if !(ENABLE(INNER_INNER_CONDITION))
            enumValueForIPCTestAPI(EnumNamespace::InnerEnumType::OtherInnerInnerValue),
#endif
        } },
#endif
#if (ENABLE(OUTER_CONDITION)) && (!(ENABLE(INNER_CONDITION)))
        { "EnumNamespace::InnerBoolType"_s, sizeof(EnumNamespace::InnerBoolType), false, {
            0, 1
        } },
#endif
    };
}

} // namespace WebKit

#endif // ENABLE(IPC_TESTING_API)
