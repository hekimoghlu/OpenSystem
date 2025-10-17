/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 3, 2021.
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

//===- TableGenBackends.h - Declarations for LLVM TableGen Backends -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations for all of the LLVM TableGen
// backends. A "TableGen backend" is just a function. See below for a
// precise description.
//
//===----------------------------------------------------------------------===//


// A TableGen backend is a function that looks like
//
//    EmitFoo(RecordKeeper &RK, raw_ostream &OS /*, anything else you need */ )
//
// What you do inside of that function is up to you, but it will usually
// involve generating C++ code to the provided raw_ostream.
//
// The RecordKeeper is just a top-level container for an in-memory
// representation of the data encoded in the TableGen file. What a TableGen
// backend does is walk around that in-memory representation and generate
// stuff based on the information it contains.
//
// The in-memory representation is a node-graph (think of it like JSON but
// with a richer ontology of types), where the nodes are subclasses of
// Record. The methods `getClass`, `getDef` are the basic interface to
// access the node-graph.  RecordKeeper also provides a handy method
// `getAllDerivedDefinitions`. Consult "include/llvm/TableGen/Record.h" for
// the exact interfaces provided by Record's and RecordKeeper.
//
// A common pattern for TableGen backends is for the EmitFoo function to
// instantiate a class which holds some context for the generation process,
// and then have most of the work happen in that class's methods. This
// pattern partly has historical roots in the previous TableGen backend API
// that involved a class and an invocation like `FooEmitter(RK).run(OS)`.
//
// Remember to wrap private things in an anonymous namespace. For most
// backends, this means that the EmitFoo function is the only thing not in
// the anonymous namespace.


// FIXME: Reorganize TableGen so that build dependencies can be more
// accurately expressed. Currently, touching any of the emitters (or
// anything that they transitively depend on) causes everything dependent
// on TableGen to be rebuilt (this includes all the targets!). Perhaps have
// a standalone TableGen binary and have the backends be loadable modules
// of some sort; then the dependency could be expressed as being on the
// module, and all the modules would have a common dependency on the
// TableGen binary with as few dependencies as possible on the rest of
// LLVM.


namespace llvm {

class raw_ostream;
class RecordKeeper;

void EmitIntrinsics(RecordKeeper &RK, raw_ostream &OS, bool TargetOnly = false);
void EmitAsmMatcher(RecordKeeper &RK, raw_ostream &OS);
void EmitAsmWriter(RecordKeeper &RK, raw_ostream &OS);
void EmitCallingConv(RecordKeeper &RK, raw_ostream &OS);
void EmitCodeEmitter(RecordKeeper &RK, raw_ostream &OS);
void EmitDAGISel(RecordKeeper &RK, raw_ostream &OS);
void EmitDFAPacketizer(RecordKeeper &RK, raw_ostream &OS);
void EmitDisassembler(RecordKeeper &RK, raw_ostream &OS);
void EmitEnhancedDisassemblerInfo(RecordKeeper &RK, raw_ostream &OS);
void EmitFastISel(RecordKeeper &RK, raw_ostream &OS);
void EmitInstrInfo(RecordKeeper &RK, raw_ostream &OS);
void EmitPseudoLowering(RecordKeeper &RK, raw_ostream &OS);
void EmitRegisterInfo(RecordKeeper &RK, raw_ostream &OS);
void EmitSubtarget(RecordKeeper &RK, raw_ostream &OS);

} // End llvm namespace
