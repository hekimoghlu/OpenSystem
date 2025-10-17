/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 7, 2023.
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

#include "CollectionScope.h"
#include <wtf/HashSet.h>
#include <wtf/Lock.h>
#include <wtf/Noncopyable.h>
#include <wtf/PrintStream.h>
#include <wtf/TZoneMalloc.h>

namespace JSC {

class CodeBlock;
class Heap;
class JSCell;
class VM;

// CodeBlockSet tracks all CodeBlocks. Every CodeBlock starts out with one
// reference coming in from GC. The GC is responsible for freeing CodeBlocks
// once they hasOneRef() and nobody is running code from that CodeBlock.

class CodeBlockSet {
    WTF_MAKE_TZONE_ALLOCATED(CodeBlockSet);
    WTF_MAKE_NONCOPYABLE(CodeBlockSet);
public:
    CodeBlockSet();
    ~CodeBlockSet();

    void mark(const AbstractLocker&, CodeBlock* candidateCodeBlock);
    
    void clearCurrentlyExecutingAndRemoveDeadCodeBlocks(VM&);

    bool contains(const AbstractLocker&, void* candidateCodeBlock);
    Lock& getLock() WTF_RETURNS_LOCK(m_lock) { return m_lock; }

    // This is expected to run only when we're not adding to the set for now. If
    // this needs to run concurrently in the future, we'll need to lock around this.
    bool isCurrentlyExecuting(CodeBlock*);

    // Visits each CodeBlock in the heap until the visitor function returns true
    // to indicate that it is done iterating, or until every CodeBlock has been
    // visited.
    template<typename Functor> void iterate(const Functor&);
    template<typename Functor> void iterate(const AbstractLocker&, const Functor&);

    template<typename Functor> void iterateCurrentlyExecuting(const Functor&);
    
    void dump(PrintStream&) const;
    
    void add(CodeBlock*);
    void remove(CodeBlock*);

private:
    UncheckedKeyHashSet<CodeBlock*> m_codeBlocks;
    UncheckedKeyHashSet<CodeBlock*> m_currentlyExecuting;
    Lock m_lock;
};

} // namespace JSC
