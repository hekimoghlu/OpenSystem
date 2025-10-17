/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 17, 2025.
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

#if !PLATFORM(COCOA)

#include "AuxiliaryProcess.h"
#include "WebKit2Initialize.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/RunLoop.h>
#include <wtf/RuntimeApplicationChecks.h>

namespace WebKit {

class AuxiliaryProcessMainCommon {
public:
    AuxiliaryProcessMainCommon();
    bool parseCommandLine(int argc, char** argv);

protected:
    AuxiliaryProcessInitializationParameters m_parameters;
};

template<typename AuxiliaryProcessType, bool HasSingleton = true>
class AuxiliaryProcessMainBase : public AuxiliaryProcessMainCommon {
public:
    virtual bool platformInitialize() { return true; }
    virtual void platformFinalize() { }

    virtual void initializeAuxiliaryProcess(AuxiliaryProcessInitializationParameters&& parameters)
    {
        if constexpr (HasSingleton)
            AuxiliaryProcessType::singleton().initialize(WTFMove(parameters));
    }

    int run(int argc, char** argv)
    {
        // setAuxiliaryProcessType() should be called before we construct
        // and initialize the AuxiliaryProcess. This is so isInXXXProcess()
        // checks are valid.
        m_parameters.processType = AuxiliaryProcessType::processType;
        setAuxiliaryProcessType(m_parameters.processType);

        if (!platformInitialize())
            return EXIT_FAILURE;

        if (!parseCommandLine(argc, argv))
            return EXIT_FAILURE;

        InitializeWebKit2();

        initializeAuxiliaryProcess(WTFMove(m_parameters));
        RunLoop::run();
        platformFinalize();

        return EXIT_SUCCESS;
    }
};

template<typename AuxiliaryProcessType>
class AuxiliaryProcessMainBaseNoSingleton : public AuxiliaryProcessMainBase<AuxiliaryProcessType, false> {
public:
    AuxiliaryProcessType& process() { return *m_process; };

    void initializeAuxiliaryProcess(AuxiliaryProcessInitializationParameters&& parameters) override
    {
        m_process = adoptRef(new AuxiliaryProcessType(WTFMove(parameters)));
    }

protected:
    RefPtr<AuxiliaryProcessType> m_process;
};

template<typename AuxiliaryProcessMainType>
int AuxiliaryProcessMain(int argc, char** argv)
{
    NeverDestroyed<AuxiliaryProcessMainType> auxiliaryMain;

    return auxiliaryMain->run(argc, argv);
}

} // namespace WebKit

#endif // !PLATFORM(COCOA)
