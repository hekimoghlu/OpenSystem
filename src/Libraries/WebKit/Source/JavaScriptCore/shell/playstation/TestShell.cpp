/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 24, 2023.
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
#include "../jsc.cpp"

#include <wtf/Language.h>

#define STATIC_OPTION(type_, name_, defaultValue_, availability_, description_) \
    static OptionsStorage::type_ orig##name_;
    FOR_EACH_JSC_OPTION(STATIC_OPTION)
#undef STATIC_OPTION

extern "C" void setupTestRun()
{
    CommandLine options(0, nullptr);
#define STATIC_OPTION(type_, name_, defaultValue_, availability_, description_) \
    orig##name_ = JSC::Options::name_();
    FOR_EACH_JSC_OPTION(STATIC_OPTION)
#undef STATIC_OPTION

    // Need to initialize WTF before we start any threads. Cannot initialize JSC
    // yet, since that would do somethings that we'd like to defer until after we
    // have a chance to parse options.
    WTF::initializeMainThread();

    // Need to override and enable restricted options before we start parsing options below.
    Config::enableRestrictedOptions();

    JSC::initialize();

    Gigacage::forbidDisablingPrimitiveGigacage();
}

extern "C" void preTest()
{
#define INIT_OPTION(type_, name_, defaultValue_, availability_, description_) \
    JSC::Options::name_() = orig##name_;
    FOR_EACH_JSC_OPTION(INIT_OPTION)
#undef INIT_OPTION
    overrideUserPreferredLanguages({ });
}

extern "C" int runTest(int argc, char* argv[])
{
    CommandLine options(argc, argv);
    processConfigFile(Options::configFile(), "jsc");

    return runJSC(
        options, true,
        [&](VM& vm, GlobalObject* globalObject, bool& success) {
            UNUSED_PARAM(vm);
            runWithOptions(globalObject, options, success);
        });
}

extern "C" void postTest()
{
}

extern "C" void shutdownTestRun()
{
}
