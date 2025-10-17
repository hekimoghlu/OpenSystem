/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 6, 2024.
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
#ifndef LLVM_C_TYPES_H
#define LLVM_C_TYPES_H

#include <IndexStoreDB_LLVMSupport/toolchain-c_DataTypes.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup LLVMCSupportTypes Types and Enumerations
 *
 * @{
 */

typedef int LLVMBool;

/* Opaque types. */

/**
 * LLVM uses a polymorphic type hierarchy which C cannot represent, therefore
 * parameters must be passed as base types. Despite the declared types, most
 * of the functions provided operate only on branches of the type hierarchy.
 * The declared parameter names are descriptive and specify which type is
 * required. Additionally, each type hierarchy is documented along with the
 * functions that operate upon it. For more detail, refer to LLVM's C++ code.
 * If in doubt, refer to Core.cpp, which performs parameter downcasts in the
 * form unwrap<RequiredType>(Param).
 */

/**
 * Used to pass regions of memory through LLVM interfaces.
 *
 * @see toolchain::MemoryBuffer
 */
typedef struct LLVMOpaqueMemoryBuffer *LLVMMemoryBufferRef;

/**
 * The top-level container for all LLVM global data. See the LLVMContext class.
 */
typedef struct LLVMOpaqueContext *LLVMContextRef;

/**
 * The top-level container for all other LLVM Intermediate Representation (IR)
 * objects.
 *
 * @see toolchain::Module
 */
typedef struct LLVMOpaqueModule *LLVMModuleRef;

/**
 * Each value in the LLVM IR has a type, an LLVMTypeRef.
 *
 * @see toolchain::Type
 */
typedef struct LLVMOpaqueType *LLVMTypeRef;

/**
 * Represents an individual value in LLVM IR.
 *
 * This models toolchain::Value.
 */
typedef struct LLVMOpaqueValue *LLVMValueRef;

/**
 * Represents a basic block of instructions in LLVM IR.
 *
 * This models toolchain::BasicBlock.
 */
typedef struct LLVMOpaqueBasicBlock *LLVMBasicBlockRef;

/**
 * Represents an LLVM Metadata.
 *
 * This models toolchain::Metadata.
 */
typedef struct LLVMOpaqueMetadata *LLVMMetadataRef;

/**
 * Represents an LLVM Named Metadata Node.
 *
 * This models toolchain::NamedMDNode.
 */
typedef struct LLVMOpaqueNamedMDNode *LLVMNamedMDNodeRef;

/**
 * Represents an entry in a Global Object's metadata attachments.
 *
 * This models std::pair<unsigned, MDNode *>
 */
typedef struct LLVMOpaqueValueMetadataEntry LLVMValueMetadataEntry;

/**
 * Represents an LLVM basic block builder.
 *
 * This models toolchain::IRBuilder.
 */
typedef struct LLVMOpaqueBuilder *LLVMBuilderRef;

/**
 * Represents an LLVM debug info builder.
 *
 * This models toolchain::DIBuilder.
 */
typedef struct LLVMOpaqueDIBuilder *LLVMDIBuilderRef;

/**
 * Interface used to provide a module to JIT or interpreter.
 * This is now just a synonym for toolchain::Module, but we have to keep using the
 * different type to keep binary compatibility.
 */
typedef struct LLVMOpaqueModuleProvider *LLVMModuleProviderRef;

/** @see toolchain::PassManagerBase */
typedef struct LLVMOpaquePassManager *LLVMPassManagerRef;

/** @see toolchain::PassRegistry */
typedef struct LLVMOpaquePassRegistry *LLVMPassRegistryRef;

/**
 * Used to get the users and usees of a Value.
 *
 * @see toolchain::Use */
typedef struct LLVMOpaqueUse *LLVMUseRef;

/**
 * Used to represent an attributes.
 *
 * @see toolchain::Attribute
 */
typedef struct LLVMOpaqueAttributeRef *LLVMAttributeRef;

/**
 * @see toolchain::DiagnosticInfo
 */
typedef struct LLVMOpaqueDiagnosticInfo *LLVMDiagnosticInfoRef;

/**
 * @see toolchain::Comdat
 */
typedef struct LLVMComdat *LLVMComdatRef;

/**
 * @see toolchain::Module::ModuleFlagEntry
 */
typedef struct LLVMOpaqueModuleFlagEntry LLVMModuleFlagEntry;

/**
 * @see toolchain::JITEventListener
 */
typedef struct LLVMOpaqueJITEventListener *LLVMJITEventListenerRef;

/**
 * @see toolchain::object::Binary
 */
typedef struct LLVMOpaqueBinary *LLVMBinaryRef;

/**
 * @}
 */

#ifdef __cplusplus
}
#endif

#endif
