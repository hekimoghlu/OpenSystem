/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 12, 2024.
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
#import "config.h"
#import "SharedMemory.h"

#import "Logging.h"
#import "ProcessIdentity.h"
#import "SharedBuffer.h"
#import <mach/mach_error.h>
#import <mach/mach_init.h>
#import <mach/mach_port.h>
#import <mach/vm_map.h>
#import <wtf/MachSendRight.h>
#import <wtf/RefPtr.h>
#import <wtf/cocoa/SpanCocoa.h>
#import <wtf/spi/cocoa/MachVMSPI.h>

#if HAVE(MACH_MEMORY_ENTRY)
#import <mach/memory_entry.h>
#endif

namespace WebCore {

#if HAVE(MACH_MEMORY_ENTRY)
static int toVMMemoryLedger(MemoryLedger memoryLedger)
{
    switch (memoryLedger) {
    case MemoryLedger::None:
        return VM_LEDGER_TAG_NONE;
    case MemoryLedger::Default:
        return VM_LEDGER_TAG_DEFAULT;
    case MemoryLedger::Network:
        return VM_LEDGER_TAG_NETWORK;
    case MemoryLedger::Media:
        return VM_LEDGER_TAG_MEDIA;
    case MemoryLedger::Graphics:
        return VM_LEDGER_TAG_GRAPHICS;
    case MemoryLedger::Neural:
        return VM_LEDGER_TAG_NEURAL;
    }
}
#endif

void SharedMemoryHandle::takeOwnershipOfMemory(MemoryLedger memoryLedger) const
{
    if (isMemoryAttributionDisabled())
        return;
#if HAVE(MACH_MEMORY_ENTRY)
    if (!m_handle)
        return;

    kern_return_t kr = mach_memory_entry_ownership(m_handle.sendRight(), mach_task_self(), toVMMemoryLedger(memoryLedger), 0);
    RELEASE_LOG_ERROR_IF(kr != KERN_SUCCESS, VirtualMemory, "SharedMemoryHandle::takeOwnershipOfMemory: Failed ownership of shared memory. Error: %" PUBLIC_LOG_STRING " (%x)", mach_error_string(kr), kr);
#else
    UNUSED_PARAM(memoryLedger);
#endif
}

void SharedMemoryHandle::setOwnershipOfMemory(const ProcessIdentity& processIdentity, MemoryLedger memoryLedger) const
{
#if HAVE(TASK_IDENTITY_TOKEN) && HAVE(MACH_MEMORY_ENTRY_OWNERSHIP_IDENTITY_TOKEN_SUPPORT)
    if (!m_handle || !processIdentity)
        return;

    kern_return_t kr = mach_memory_entry_ownership(m_handle.sendRight(), processIdentity.taskIdToken(), toVMMemoryLedger(memoryLedger), 0);
    RELEASE_LOG_ERROR_IF(kr != KERN_SUCCESS, VirtualMemory, "SharedMemoryHandle::setOwnershipOfMemory: Failed ownership of shared memory. Error: %" PUBLIC_LOG_STRING " (%x)", mach_error_string(kr), kr);
#else
    UNUSED_PARAM(memoryLedger);
    UNUSED_PARAM(processIdentity);
#endif
}

static inline void* toPointer(mach_vm_address_t address)
{
    return reinterpret_cast<void*>(static_cast<uintptr_t>(address));
}

static inline mach_vm_address_t toVMAddress(void* pointer)
{
    return static_cast<mach_vm_address_t>(reinterpret_cast<uintptr_t>(pointer));
}

RefPtr<SharedMemory> SharedMemory::allocate(size_t size)
{
    ASSERT(size);

    mach_vm_address_t address = 0;
    // Using VM_FLAGS_PURGABLE so that we can later transfer ownership of the memory via mach_memory_entry_ownership().
    kern_return_t kr = mach_vm_allocate(mach_task_self(), &address, size, VM_FLAGS_ANYWHERE | VM_FLAGS_PURGABLE);
    if (kr != KERN_SUCCESS) {
        RELEASE_LOG_ERROR(VirtualMemory, "%p - SharedMemory::allocate: Failed to allocate mach_vm_allocate shared memory (%zu bytes). %" PUBLIC_LOG_STRING " (%x)", nullptr, size, mach_error_string(kr), kr);
        return nullptr;
    }

    Ref sharedMemory = adoptRef(*new SharedMemory);
    sharedMemory->m_size = size;
    sharedMemory->m_data = toPointer(address);
    sharedMemory->m_protection = Protection::ReadWrite;
    return WTFMove(sharedMemory);
}

static inline vm_prot_t machProtection(SharedMemory::Protection protection)
{
    switch (protection) {
    case SharedMemory::Protection::ReadOnly:
        return VM_PROT_READ;
    case SharedMemory::Protection::ReadWrite:
        return VM_PROT_READ | VM_PROT_WRITE;
    }

    ASSERT_NOT_REACHED();
    return VM_PROT_NONE;
}

static MachSendRight makeMemoryEntry(size_t size, vm_offset_t offset, SharedMemory::Protection protection, mach_port_t parentEntry)
{
    memory_object_size_t memoryObjectSize = size;
    mach_port_t port = MACH_PORT_NULL;

#if HAVE(MEMORY_ATTRIBUTION_VM_SHARE_SUPPORT)
    kern_return_t kr = mach_make_memory_entry_64(mach_task_self(), &memoryObjectSize, offset, machProtection(protection) | VM_PROT_IS_MASK | MAP_MEM_VM_SHARE | MAP_MEM_USE_DATA_ADDR, &port, parentEntry);
    if (kr != KERN_SUCCESS) {
        RELEASE_LOG_ERROR(VirtualMemory, "SharedMemory::makeMemoryEntry: Failed to create a mach port for shared memory. Error: %" PUBLIC_LOG_STRING " (%x)", mach_error_string(kr), kr);
        return { };
    }
#else
    // First try without MAP_MEM_VM_SHARE because it prevents memory ownership transfer. We only pass the MAP_MEM_VM_SHARE flag as a fallback.
    kern_return_t kr = mach_make_memory_entry_64(mach_task_self(), &memoryObjectSize, offset, machProtection(protection) | VM_PROT_IS_MASK | MAP_MEM_USE_DATA_ADDR, &port, parentEntry);
    if (kr != KERN_SUCCESS) {
        RELEASE_LOG(VirtualMemory, "SharedMemory::makeMemoryEntry: Failed to create a mach port for shared memory, will try again with MAP_MEM_VM_SHARE flag. Error: %" PUBLIC_LOG_STRING " (%x)", mach_error_string(kr), kr);
        kr = mach_make_memory_entry_64(mach_task_self(), &memoryObjectSize, offset, machProtection(protection) | VM_PROT_IS_MASK | MAP_MEM_VM_SHARE | MAP_MEM_USE_DATA_ADDR, &port, parentEntry);
        if (kr != KERN_SUCCESS) {
            RELEASE_LOG_ERROR(VirtualMemory, "SharedMemory::makeMemoryEntry: Failed to create a mach port for shared memory with MAP_MEM_VM_SHARE flag. Error: %" PUBLIC_LOG_STRING " (%x)", mach_error_string(kr), kr);
            return { };
        }
    }
#endif // HAVE(MEMORY_ATTRIBUTION_VM_SHARE_SUPPORT)

    RELEASE_ASSERT(memoryObjectSize >= size);

    return MachSendRight::adopt(port);
}

RefPtr<SharedMemory> SharedMemory::wrapMap(std::span<const uint8_t> data, Protection protection)
{
    ASSERT(!data.empty());

    auto sendRight = makeMemoryEntry(data.size(), toVMAddress(const_cast<uint8_t*>(data.data())), protection, MACH_PORT_NULL);
    if (!sendRight)
        return nullptr;

    Ref sharedMemory = adoptRef(*new SharedMemory);
    sharedMemory->m_size = data.size();
    sharedMemory->m_data = nullptr;
    sharedMemory->m_sendRight = WTFMove(sendRight);
    sharedMemory->m_protection = protection;

    return WTFMove(sharedMemory);
}

RefPtr<SharedMemory> SharedMemory::map(Handle&& handle, Protection protection)
{
    vm_prot_t vmProtection = machProtection(protection);
    mach_vm_address_t mappedAddress = 0;

    kern_return_t kr = mach_vm_map(mach_task_self(), &mappedAddress, handle.m_size, 0, VM_FLAGS_ANYWHERE | VM_FLAGS_RETURN_DATA_ADDR, handle.m_handle.sendRight(), 0, false, vmProtection, vmProtection, VM_INHERIT_NONE);
    if (kr != KERN_SUCCESS) {
        RELEASE_LOG_ERROR(VirtualMemory, "%p - SharedMemory::map: Failed to map shared memory. %" PUBLIC_LOG_STRING " (%x)", nullptr, mach_error_string(kr), kr);
        return nullptr;
    }

    Ref sharedMemory = adoptRef(*new SharedMemory);
    sharedMemory->m_size = handle.m_size;
    sharedMemory->m_data = toPointer(mappedAddress);
    sharedMemory->m_protection = protection;

    return WTFMove(sharedMemory);
}

SharedMemory::~SharedMemory()
{
    if (m_data) {
        kern_return_t kr = mach_vm_deallocate(mach_task_self(), toVMAddress(m_data), m_size);
        if (kr != KERN_SUCCESS) {
            RELEASE_LOG_ERROR(VirtualMemory, "%p - SharedMemory::~SharedMemory: Failed to deallocate shared memory. %" PUBLIC_LOG_STRING " (%x)", this, mach_error_string(kr), kr);
            ASSERT_NOT_REACHED();
        }
    }
}

auto SharedMemory::createHandle(Protection protection) -> std::optional<Handle>
{
    auto sendRight = createSendRight(protection);
    if (!sendRight)
        return std::nullopt;
    return { Handle(WTFMove(sendRight), m_size) };
}

WTF::MachSendRight SharedMemory::createSendRight(Protection protection) const
{
    ASSERT(m_protection == protection || m_protection == Protection::ReadWrite && protection == Protection::ReadOnly);
    ASSERT(!!m_data ^ !!m_sendRight);

    if (m_sendRight && m_protection == protection)
        return MachSendRight { m_sendRight };

    ASSERT(m_data);
    return makeMemoryEntry(m_size, toVMAddress(m_data), protection, MACH_PORT_NULL);
}

RetainPtr<NSData> SharedMemory::toNSData() const
{
    return WTF::toNSData(span());
}

} // namespace WebCore
