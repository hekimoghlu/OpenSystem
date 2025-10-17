/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 6, 2025.
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
#ifndef SandboxInitializationParameters_h
#define SandboxInitializationParameters_h

#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

#if PLATFORM(COCOA)
OBJC_CLASS NSString;
#endif

namespace WebKit {

class SandboxInitializationParameters {
    WTF_MAKE_NONCOPYABLE(SandboxInitializationParameters);
public:
    SandboxInitializationParameters();
    ~SandboxInitializationParameters();

#if PLATFORM(COCOA)
    void addConfDirectoryParameter(ASCIILiteral name, int confID);
    void addPathParameter(ASCIILiteral name, NSString *path);
    void addPathParameter(ASCIILiteral name, const char* path);
    void addParameter(ASCIILiteral name, CString&& value);

    Vector<const char*> namedParameterVector() const;

    size_t count() const;
    ASCIILiteral name(size_t index) const;
    const char* value(size_t index) const;

    enum class ProfileSelectionMode : uint8_t {
        UseDefaultSandboxProfilePath,
        UseOverrideSandboxProfilePath,
        UseSandboxProfile
    };

    ProfileSelectionMode mode() const { return m_profileSelectionMode; }

    void setOverrideSandboxProfilePath(const String& path)
    {
        m_profileSelectionMode = ProfileSelectionMode::UseOverrideSandboxProfilePath;
        m_overrideSandboxProfilePathOrSandboxProfile = path;
    }

    const String& overrideSandboxProfilePath() const
    {
        ASSERT(m_profileSelectionMode == ProfileSelectionMode::UseOverrideSandboxProfilePath);
        return m_overrideSandboxProfilePathOrSandboxProfile;
    }

    void setSandboxProfile(const String& profile)
    {
        m_profileSelectionMode = ProfileSelectionMode::UseSandboxProfile;
        m_overrideSandboxProfilePathOrSandboxProfile = profile;
    }

    const String& sandboxProfile() const
    {
        ASSERT(m_profileSelectionMode == ProfileSelectionMode::UseSandboxProfile);
        return m_overrideSandboxProfilePathOrSandboxProfile;
    }

    void setUserDirectorySuffix(const String& suffix) { m_userDirectorySuffix = suffix; }
    const String& userDirectorySuffix() const { return m_userDirectorySuffix; }
#endif

private:
#if PLATFORM(COCOA)
    void appendPathInternal(ASCIILiteral name, const char* path);

    mutable Vector<ASCIILiteral> m_parameterNames;
    mutable Vector<CString> m_parameterValues;
    String m_userDirectorySuffix;

    ProfileSelectionMode m_profileSelectionMode;
    String m_overrideSandboxProfilePathOrSandboxProfile;
#endif
};

#if !PLATFORM(COCOA)
SandboxInitializationParameters::SandboxInitializationParameters()
{
}

SandboxInitializationParameters::~SandboxInitializationParameters()
{
}
#endif

} // namespace WebKit

#endif // SandboxInitializationParameters_h
