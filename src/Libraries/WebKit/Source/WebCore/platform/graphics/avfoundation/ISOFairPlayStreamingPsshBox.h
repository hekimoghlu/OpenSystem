/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 29, 2024.
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

#include "ISOProtectionSystemSpecificHeaderBox.h"

namespace WebCore {

class ISOFairPlayStreamingInfoBox final : public ISOFullBox {
public:
    WEBCORE_EXPORT ISOFairPlayStreamingInfoBox();
    WEBCORE_EXPORT ISOFairPlayStreamingInfoBox(const ISOFairPlayStreamingInfoBox&);
    ISOFairPlayStreamingInfoBox(ISOFairPlayStreamingInfoBox&&);
    WEBCORE_EXPORT ~ISOFairPlayStreamingInfoBox();

    static FourCC boxTypeName() { return std::span { "fpsi" }; }

    FourCC scheme() const { return m_scheme; }

    WEBCORE_EXPORT bool parse(JSC::DataView&, unsigned& offset) final;

private:
    FourCC m_scheme;
};

class ISOFairPlayStreamingKeyRequestInfoBox final : public ISOFullBox {
public:
    WEBCORE_EXPORT ISOFairPlayStreamingKeyRequestInfoBox();
    WEBCORE_EXPORT ~ISOFairPlayStreamingKeyRequestInfoBox();

    static FourCC boxTypeName() { return std::span { "fkri" }; }

    using KeyID = Vector<uint8_t, 16>;
    const KeyID& keyID() const { return m_keyID; }

    WEBCORE_EXPORT bool parse(JSC::DataView&, unsigned& offset) final;

private:
    KeyID m_keyID;
};

class ISOFairPlayStreamingKeyAssetIdBox final : public ISOBox {
public:
    WEBCORE_EXPORT ISOFairPlayStreamingKeyAssetIdBox();
    WEBCORE_EXPORT ISOFairPlayStreamingKeyAssetIdBox(const ISOFairPlayStreamingKeyAssetIdBox&);
    ISOFairPlayStreamingKeyAssetIdBox(ISOFairPlayStreamingKeyAssetIdBox&&) = default;
    WEBCORE_EXPORT ~ISOFairPlayStreamingKeyAssetIdBox();

    ISOFairPlayStreamingKeyAssetIdBox& operator=(const ISOFairPlayStreamingKeyAssetIdBox&) = default;
    ISOFairPlayStreamingKeyAssetIdBox& operator=(ISOFairPlayStreamingKeyAssetIdBox&&) = default;

    static FourCC boxTypeName() { return std::span { "fkai" }; }
    const Vector<uint8_t> data() const { return m_data; }

    WEBCORE_EXPORT bool parse(JSC::DataView&, unsigned& offset) final;

private:
    Vector<uint8_t> m_data;
};

class ISOFairPlayStreamingKeyContextBox final : public ISOBox {
public:
    WEBCORE_EXPORT ISOFairPlayStreamingKeyContextBox();
    WEBCORE_EXPORT ISOFairPlayStreamingKeyContextBox(const ISOFairPlayStreamingKeyContextBox&);
    ISOFairPlayStreamingKeyContextBox(ISOFairPlayStreamingKeyContextBox&&) = default;
    WEBCORE_EXPORT ~ISOFairPlayStreamingKeyContextBox();

    ISOFairPlayStreamingKeyContextBox& operator=(const ISOFairPlayStreamingKeyContextBox&) = default;
    ISOFairPlayStreamingKeyContextBox& operator=(ISOFairPlayStreamingKeyContextBox&&) = default;

    static FourCC boxTypeName() { return std::span { "fkcx" }; }
    const Vector<uint8_t> data() const { return m_data; }

    WEBCORE_EXPORT bool parse(JSC::DataView&, unsigned& offset) final;

private:
    Vector<uint8_t> m_data;
};

class ISOFairPlayStreamingKeyVersionListBox final : public ISOBox {
public:
    WEBCORE_EXPORT ISOFairPlayStreamingKeyVersionListBox();
    WEBCORE_EXPORT ISOFairPlayStreamingKeyVersionListBox(const ISOFairPlayStreamingKeyVersionListBox&);
    ISOFairPlayStreamingKeyVersionListBox(ISOFairPlayStreamingKeyVersionListBox&&) = default;
    WEBCORE_EXPORT ~ISOFairPlayStreamingKeyVersionListBox();

    ISOFairPlayStreamingKeyVersionListBox& operator=(const ISOFairPlayStreamingKeyVersionListBox&) = default;
    ISOFairPlayStreamingKeyVersionListBox& operator=(ISOFairPlayStreamingKeyVersionListBox&&) = default;

    static FourCC boxTypeName() { return std::span { "fkvl" }; }
    const Vector<uint8_t> versions() const { return m_versions; }

    WEBCORE_EXPORT bool parse(JSC::DataView&, unsigned& offset) final;

private:
    Vector<uint8_t> m_versions;
};

class ISOFairPlayStreamingKeyRequestBox final : public ISOBox {
public:
    WEBCORE_EXPORT ISOFairPlayStreamingKeyRequestBox();
    WEBCORE_EXPORT ISOFairPlayStreamingKeyRequestBox(const ISOFairPlayStreamingKeyRequestBox&);
    ISOFairPlayStreamingKeyRequestBox(ISOFairPlayStreamingKeyRequestBox&&) = default;
    WEBCORE_EXPORT ~ISOFairPlayStreamingKeyRequestBox();

    static FourCC boxTypeName() { return std::span { "fpsk" }; }

    const ISOFairPlayStreamingKeyRequestInfoBox& requestInfo() const { return m_requestInfo; }
    const std::optional<ISOFairPlayStreamingKeyAssetIdBox>& assetID() const { return m_assetID; }
    const std::optional<ISOFairPlayStreamingKeyContextBox>& content() const { return m_context; }
    const std::optional<ISOFairPlayStreamingKeyVersionListBox>& versionList() const { return m_versionList; }

    WEBCORE_EXPORT bool parse(JSC::DataView&, unsigned& offset) final;

private:
    ISOFairPlayStreamingKeyRequestInfoBox m_requestInfo;
    std::optional<ISOFairPlayStreamingKeyAssetIdBox> m_assetID;
    std::optional<ISOFairPlayStreamingKeyContextBox> m_context;
    std::optional<ISOFairPlayStreamingKeyVersionListBox> m_versionList;
};

class ISOFairPlayStreamingInitDataBox final : public ISOBox {
public:
    WEBCORE_EXPORT ISOFairPlayStreamingInitDataBox();
    WEBCORE_EXPORT ~ISOFairPlayStreamingInitDataBox();

    static FourCC boxTypeName() { return std::span { "fpsd" }; }

    const ISOFairPlayStreamingInfoBox& info() const { return m_info; }
    const Vector<ISOFairPlayStreamingKeyRequestBox>& requests() const { return m_requests; }

    WEBCORE_EXPORT bool parse(JSC::DataView&, unsigned& offset) final;

private:
    ISOFairPlayStreamingInfoBox m_info;
    Vector<ISOFairPlayStreamingKeyRequestBox> m_requests;
};

class ISOFairPlayStreamingPsshBox final : public ISOProtectionSystemSpecificHeaderBox {
public:
    WEBCORE_EXPORT ISOFairPlayStreamingPsshBox();
    WEBCORE_EXPORT ~ISOFairPlayStreamingPsshBox();

    static const Vector<uint8_t>& fairPlaySystemID();

    const ISOFairPlayStreamingInitDataBox& initDataBox() { return m_initDataBox; }

    WEBCORE_EXPORT bool parse(JSC::DataView&, unsigned& offset) final;

private:
    ISOFairPlayStreamingInitDataBox m_initDataBox;
};

}

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::ISOFairPlayStreamingPsshBox) \
static bool isType(const WebCore::ISOProtectionSystemSpecificHeaderBox& psshBox) { return psshBox.systemID() == WebCore::ISOFairPlayStreamingPsshBox::fairPlaySystemID(); }
SPECIALIZE_TYPE_TRAITS_END()
