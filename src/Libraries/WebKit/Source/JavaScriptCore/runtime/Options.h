/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 7, 2023.
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

#include "CPU.h"
#include "JSCConfig.h"
#include "JSExportMacros.h"
#include <stdint.h>
#include <wtf/ForbidHeapAllocation.h>
#include <wtf/Noncopyable.h>
#include <wtf/PrintStream.h>
#include <wtf/StdLibExtras.h>

namespace WTF {
class StringBuilder;
}
using WTF::StringBuilder;

namespace JSC {

class Options {
public:
    enum class SandboxPolicy : uint8_t {
        Unknown,
        Allow,
        Block,
    };

    enum class DumpLevel : uint8_t {
        None = 0,
        Overridden,
        All,
        Verbose
    };

    enum class Availability : uint8_t  {
        Normal = 0,
        Restricted,
        Configurable
    };

#define DECLARE_OPTION_ID(type_, name_, defaultValue_, availability_, description_) \
    name_##ID,

    enum ID : uint16_t {
        FOR_EACH_JSC_OPTION(DECLARE_OPTION_ID)
    };
#undef DECLARE_OPTION_ID

    enum class Type : uint8_t {
        Bool,
        Unsigned,
        Double,
        Int32,
        Size,
        OptionRange,
        OptionString,
        GCLogLevel,
        OSLogType,
    };

    class AllowUnfinalizedAccessScope {
        WTF_MAKE_NONCOPYABLE(AllowUnfinalizedAccessScope);
        WTF_FORBID_HEAP_ALLOCATION;
    public:
#if ASSERT_ENABLED
        AllowUnfinalizedAccessScope()
        {
            if (!g_jscConfig.options.isFinalized) {
                m_savedAllowUnfinalizedUse = g_jscConfig.options.allowUnfinalizedAccess;
                g_jscConfig.options.allowUnfinalizedAccess = true;
            }
        }

        ~AllowUnfinalizedAccessScope()
        {
            if (!g_jscConfig.options.isFinalized)
                g_jscConfig.options.allowUnfinalizedAccess = m_savedAllowUnfinalizedUse;
        }

    private:
        bool m_savedAllowUnfinalizedUse;
#else
        ALWAYS_INLINE AllowUnfinalizedAccessScope() = default;
        ALWAYS_INLINE ~AllowUnfinalizedAccessScope() { }
#endif
    };

    JS_EXPORT_PRIVATE static void initialize();
    static void finalize();

    JS_EXPORT_PRIVATE static bool setAllJITCodeValidations(const char* arg);

    // Parses a string of options where each option is of the format "--<optionName>=<value>"
    // and are separated by a space. The leading "--" is optional and will be ignored.
    JS_EXPORT_PRIVATE static bool setOptions(const char* optionsList);

    // Parses a single command line option in the format "<optionName>=<value>"
    // (no spaces allowed) and set the specified option if appropriate.
    JS_EXPORT_PRIVATE static bool setOption(const char* arg, bool verify = true);

    JS_EXPORT_PRIVATE static void executeDumpOptions();
    JS_EXPORT_PRIVATE static void dumpAllOptionsInALine(StringBuilder&);

    JS_EXPORT_PRIVATE static void assertOptionsAreCoherent();
    JS_EXPORT_PRIVATE static void notifyOptionsChanged();

#define DECLARE_OPTION_ACCESSORS(type_, name_, defaultValue_, availability_, description_) \
public: \
    ALWAYS_INLINE static OptionsStorage::type_& name_() \
    { \
        ASSERT(g_jscConfig.options.allowUnfinalizedAccess || g_jscConfig.options.isFinalized); \
        return g_jscConfig.options.name_; \
    }

    FOR_EACH_JSC_OPTION(DECLARE_OPTION_ACCESSORS)
#undef DECLARE_OPTION_ACCESSORS

    static bool isAvailable(ID, Availability);
    JS_EXPORT_PRIVATE static SandboxPolicy machExceptionHandlerSandboxPolicy;

private:
    Options();

    enum DumpDefaultsOption {
        DontDumpDefaults,
        DumpDefaults
    };
    static void dumpAllOptions(DumpLevel, ASCIILiteral title = { });
    static void dumpAllOptions(StringBuilder&, DumpLevel, ASCIILiteral title,
        ASCIILiteral separator, ASCIILiteral optionHeader, ASCIILiteral optionFooter, DumpDefaultsOption);
    static void dumpOption(StringBuilder&, DumpLevel, ID,
        ASCIILiteral optionHeader, ASCIILiteral optionFooter, DumpDefaultsOption);

    static bool setOptionWithoutAlias(const char* arg, bool verify = true);
    static bool setAliasedOption(const char* arg, bool verify = true);
#if !PLATFORM(COCOA)
    static bool overrideAliasedOptionWithHeuristic(const char* name);
#endif

    static void setAllJITCodeValidations(bool);

    static bool defaultTCSMValue();
    static unsigned computeNumberOfGCMarkers(unsigned maxNumberOfGCMarkers);
    static unsigned computeNumberOfWorkerThreads(int maxNumberOfWorkerThreads, int minimum = 1);
    static int32_t computePriorityDeltaOfWorkerThreads(int32_t twoCorePriorityDelta, int32_t multiCorePriorityDelta);
    static constexpr bool jitEnabledByDefault() { return is32Bit() || isAddress64Bit(); }
    static constexpr bool ipintEnabledByDefault() { return isARM64() || isARM64E() || isX86_64(); }
};

} // namespace JSC
