/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 8, 2025.
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

#if ENABLE(JIT)

#include <wtf/Vector.h>

namespace JSC {

class JITPlan;
class Scannable;
class VM;

class Safepoint {
public:
    class Result {
    public:
        Result()
            : m_didGetCancelled(false)
            , m_wasChecked(true)
        {
        }
        
        ~Result();
        
        bool didGetCancelled();
        
    private:
        friend class Safepoint;
        
        bool m_didGetCancelled;
        bool m_wasChecked;
        bool m_keepDependenciesLive;
    };
    
    Safepoint(JITPlan&, Result&);
    ~Safepoint();
    
    void add(Scannable*);
    
    void begin(bool keepDependenciesLive);

    template<typename Visitor> void checkLivenessAndVisitChildren(Visitor&);
    template<typename Visitor> bool isKnownToBeLiveDuringGC(Visitor&);
    bool isKnownToBeLiveAfterGC();
    void cancel();
    bool keepDependenciesLive() const;
    
    VM* vm() const; // May return null if we've been cancelled.

private:
    VM* m_vm;
    JITPlan& m_plan;
    Vector<Scannable*> m_scannables;
    bool m_didCallBegin;
    bool m_keepDependenciesLive;
    Result& m_result;
};

} // namespace JSC

#endif // ENABLE(JIT)
