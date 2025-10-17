/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 28, 2024.
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

//===--- IndexStoreCXX.h - C++ wrapper for the Index Store C API. ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// A version of IndexStoreCXX.h that uses indexstore_functions.h
//
//===----------------------------------------------------------------------===//

#ifndef INDEXSTOREDB_INDEXSTORE_INDEXSTORECXX_H
#define INDEXSTOREDB_INDEXSTORE_INDEXSTORECXX_H

#include <IndexStoreDB_Index/indexstore_functions.h>
#include <IndexStoreDB_LLVMSupport/toolchain_ADT_ArrayRef.h>
#include <IndexStoreDB_LLVMSupport/toolchain_ADT_Optional.h>
#include <IndexStoreDB_LLVMSupport/toolchain_ADT_STLExtras.h>
#include <IndexStoreDB_LLVMSupport/toolchain_ADT_SmallString.h>
#include <IndexStoreDB_LLVMSupport/toolchain_Support_Mutex.h>
#include <ctime>
#include <vector>

namespace indexstore {
  using toolchain::ArrayRef;
  using toolchain::Optional;
  using toolchain::StringRef;

static inline StringRef stringFromIndexStoreStringRef(indexstore_string_ref_t str) {
  return StringRef(str.data, str.length);
}

class IndexStoreLibrary;
typedef std::shared_ptr<IndexStoreLibrary> IndexStoreLibraryRef;

class IndexStoreLibrary {
  indexstore_functions_t functions;
public:
  IndexStoreLibrary(indexstore_functions_t functions) : functions(functions) {}

  const indexstore_functions_t &api() const { return functions; }
};

class IndexStoreCreationOptions {
  std::vector<std::pair<std::string, std::string>> prefixMap;

public:
  IndexStoreCreationOptions() {}

  void addPrefixMapping(StringRef orig, StringRef remapped) {
    prefixMap.emplace_back(std::string(orig), std::string(remapped));
  }

  bool hasPrefixMappings() const { return !prefixMap.empty(); }

  const std::vector<std::pair<std::string, std::string>> &getPrefixMap() const { return prefixMap; }

  /// Convert this into a `indexstore_creation_options_t` that the caller
  /// owns and is responsible for disposing.
  indexstore_creation_options_t createOptions(const indexstore_functions_t &api) const {
    indexstore_creation_options_t options = api.creation_options_create();
    for (const auto &Mapping : prefixMap) {
      api.creation_options_add_prefix_mapping(options, Mapping.first.c_str(),
                                              Mapping.second.c_str());
    }
    return options;
  }
};

template<typename FnT, typename ...Params>
static inline auto functionPtrFromFunctionRef(void *ctx, Params ...params)
    -> decltype((*(FnT *)ctx)(std::forward<Params>(params)...)) {
  auto fn = (FnT *)ctx;
  return (*fn)(std::forward<Params>(params)...);
}

class IndexRecordSymbol {
  indexstore_symbol_t obj;
  IndexStoreLibraryRef lib;
  friend class IndexRecordReader;

public:
  IndexRecordSymbol(indexstore_symbol_t obj, IndexStoreLibraryRef lib) : obj(obj), lib(std::move(lib)) {}

  indexstore_symbol_language_t getLanguage() {
    return lib->api().symbol_get_language(obj);
  }
  indexstore_symbol_kind_t getKind() { return lib->api().symbol_get_kind(obj); }
  indexstore_symbol_subkind_t getSubKind() { return lib->api().symbol_get_subkind(obj); }
  uint64_t getProperties() {
    return lib->api().symbol_get_properties(obj);
  }
  uint64_t getRoles() { return lib->api().symbol_get_roles(obj); }
  uint64_t getRelatedRoles() { return lib->api().symbol_get_related_roles(obj); }
  StringRef getName() { return stringFromIndexStoreStringRef(lib->api().symbol_get_name(obj)); }
  StringRef getUSR() { return stringFromIndexStoreStringRef(lib->api().symbol_get_usr(obj)); }
  StringRef getCodegenName() { return stringFromIndexStoreStringRef(lib->api().symbol_get_codegen_name(obj)); }
};

class IndexSymbolRelation {
  indexstore_symbol_relation_t obj;
  IndexStoreLibraryRef lib;

public:
  IndexSymbolRelation(indexstore_symbol_relation_t obj, IndexStoreLibraryRef lib) : obj(obj), lib(std::move(lib)) {}

  uint64_t getRoles() { return lib->api().symbol_relation_get_roles(obj); }
  IndexRecordSymbol getSymbol() { return {lib->api().symbol_relation_get_symbol(obj), lib}; }
};

class IndexRecordOccurrence {
  indexstore_occurrence_t obj;
  IndexStoreLibraryRef lib;

public:
  IndexRecordOccurrence(indexstore_occurrence_t obj, IndexStoreLibraryRef lib) : obj(obj), lib(std::move(lib)) {}

  IndexRecordSymbol getSymbol() { return {lib->api().occurrence_get_symbol(obj), lib}; }
  uint64_t getRoles() { return lib->api().occurrence_get_roles(obj); }

  bool foreachRelation(toolchain::function_ref<bool(IndexSymbolRelation)> receiver) {
    auto forwarder = [&](indexstore_symbol_relation_t sym_rel) -> bool {
      return receiver({sym_rel, lib});
    };
    return lib->api().occurrence_relations_apply_f(obj, &forwarder, functionPtrFromFunctionRef<decltype(forwarder)>);
  }

  std::pair<unsigned, unsigned> getLineCol() {
    unsigned line, col;
    lib->api().occurrence_get_line_col(obj, &line, &col);
    return std::make_pair(line, col);
  }
};

class IndexStore;
typedef std::shared_ptr<IndexStore> IndexStoreRef;

class IndexStore {
  indexstore_t obj;
  IndexStoreLibraryRef library;
  friend class IndexRecordReader;
  friend class IndexUnitReader;

public:
  IndexStore(StringRef path, IndexStoreLibraryRef library,
             const IndexStoreCreationOptions &options,
             std::string &error) : library(std::move(library)) {
    toolchain::SmallString<64> buf = path;
    indexstore_error_t c_err = nullptr;
    // Backwards compatibility for previous versions which don't support a
    // prefix mapping.
    if (api().store_create_with_options) {
      indexstore_creation_options_t c_options = options.createOptions(api());
      obj = api().store_create_with_options(buf.c_str(), c_options, &c_err);
      api().creation_options_dispose(c_options);
    } else if (options.hasPrefixMappings()) {
      error = "Prefix mappings unavailable in this version of libIndexStore.";
      obj = nullptr;
      return;
    } else {
      obj = api().store_create(buf.c_str(), &c_err);
    }
    if (c_err) {
      error = api().error_get_description(c_err);
      api().error_dispose(c_err);
    }
  }

  IndexStore(IndexStore &&other) : obj(other.obj) {
    other.obj = nullptr;
  }

  ~IndexStore() {
    api().store_dispose(obj);
  }

  static IndexStoreRef create(StringRef path, IndexStoreLibraryRef library,
                              const IndexStoreCreationOptions &options,
                              std::string &error) {
    auto storeRef = std::make_shared<IndexStore>(path, std::move(library), options, error);
    if (storeRef->isInvalid())
      return nullptr;
    return storeRef;
  }

  const indexstore_functions_t &api() const { return library->api(); }

  unsigned formatVersion() {
    return api().format_version();
  }

  bool isValid() const { return obj; }
  bool isInvalid() const { return !isValid(); }
  explicit operator bool() const { return isValid(); }

  bool foreachUnit(bool sorted, toolchain::function_ref<bool(StringRef unitName)> receiver) {
    auto forwarder = [&](indexstore_string_ref_t unit_name) -> bool {
      return receiver(stringFromIndexStoreStringRef(unit_name));
    };
    return api().store_units_apply_f(obj, sorted, &forwarder, functionPtrFromFunctionRef<decltype(forwarder)>);
  }

  class UnitEvent {
    indexstore_unit_event_t obj;
    IndexStoreLibraryRef lib;
  public:
    UnitEvent(indexstore_unit_event_t obj, IndexStoreLibraryRef lib) : obj(obj), lib(std::move(lib)) {}

    enum class Kind {
      Added,
      Removed,
      Modified,
      DirectoryDeleted,
    };
    Kind getKind() const {
      indexstore_unit_event_kind_t c_k = lib->api().unit_event_get_kind(obj);
      Kind K;
      switch (c_k) {
      case INDEXSTORE_UNIT_EVENT_ADDED: K = Kind::Added; break;
      case INDEXSTORE_UNIT_EVENT_REMOVED: K = Kind::Removed; break;
      case INDEXSTORE_UNIT_EVENT_MODIFIED: K = Kind::Modified; break;
      case INDEXSTORE_UNIT_EVENT_DIRECTORY_DELETED: K = Kind::DirectoryDeleted; break;
      }
      return K;
    }

    StringRef getUnitName() const {
      return stringFromIndexStoreStringRef(lib->api().unit_event_get_unit_name(obj));
    }
  };

  class UnitEventNotification {
    indexstore_unit_event_notification_t obj;
    IndexStoreLibraryRef lib;
  public:
    UnitEventNotification(indexstore_unit_event_notification_t obj, IndexStoreLibraryRef lib) : obj(obj), lib(std::move(lib)) {}

    bool isInitial() const { return lib->api().unit_event_notification_is_initial(obj); }
    size_t getEventsCount() const { return lib->api().unit_event_notification_get_events_count(obj); }
    UnitEvent getEvent(size_t index) const {
      return UnitEvent{lib->api().unit_event_notification_get_event(obj, index), lib};
    }
  };

  typedef std::function<void(UnitEventNotification)> UnitEventHandler;
  typedef std::function<void(indexstore_unit_event_notification_t)> RawUnitEventHandler;

private:
  struct EventHandlerContext {
    RawUnitEventHandler handler;
    #if __has_feature(thread_sanitizer)
    std::unique_ptr<toolchain::sys::Mutex> eventHandlerMutex;
    #endif

    EventHandlerContext(RawUnitEventHandler handler) : handler(std::move(handler)) {
      #if __has_feature(thread_sanitizer)
      eventHandlerMutex = std::make_unique<toolchain::sys::Mutex>();
      #endif
    }

    ~EventHandlerContext() {
      #if __has_feature(thread_sanitizer)
      // See comment in event_handler_finalizer.
      assert(!eventHandlerMutex);
      #endif
    }
  };

public:

  void setUnitEventHandler(UnitEventHandler handler) {
    auto localLib = std::weak_ptr<IndexStoreLibrary>(library);
    if (!handler) {
      api().store_set_unit_event_handler_f(obj, nullptr, nullptr, nullptr);
      return;
    }

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc++14-extensions"
    auto fnPtr = new EventHandlerContext([handler, localLib=std::move(localLib)](
        indexstore_unit_event_notification_t evt_note) {
      if (auto lib = localLib.lock()) {
        handler(UnitEventNotification(evt_note, lib));
      }
    });
#pragma clang diagnostic pop
    api().store_set_unit_event_handler_f(obj, fnPtr, event_handler, event_handler_finalizer);
  }

private:
  static void event_handler(void *ctx_, indexstore_unit_event_notification_t evt) {
    auto ctx = (EventHandlerContext*)ctx_;

    #if __has_feature(thread_sanitizer)
    // See comment in event_handler_finalizer.
    toolchain::sys::ScopedLock L(*ctx->eventHandlerMutex);
    #endif

    (ctx->handler)(evt);
  }
  static void event_handler_finalizer(void *ctx_) {
    auto ctx = (EventHandlerContext*)ctx_;

    #if __has_feature(thread_sanitizer)
    // We need to convince TSan that the event handler callback never overlaps the
    // destructor of the context. We use `eventHandlerMutex` to ensure TSan can
    // see the synchronization, and we need to move the mutex out of the context
    // so that it can be held during `delete` below.
    auto mutexPtr = std::move(ctx->eventHandlerMutex);
    toolchain::sys::ScopedLock L(*mutexPtr);
    #endif

    delete ctx;
  }

public:
  bool startEventListening(bool waitInitialSync, std::string &error) {
    indexstore_unit_event_listen_options_t opts;
    opts.wait_initial_sync = waitInitialSync;
    indexstore_error_t c_err = nullptr;
    bool ret = api().store_start_unit_event_listening(obj, &opts, sizeof(opts), &c_err);
    if (c_err) {
      error = api().error_get_description(c_err);
      api().error_dispose(c_err);
    }
    return ret;
  }

  void stopEventListening() {
    return api().store_stop_unit_event_listening(obj);
  }

  void discardUnit(StringRef UnitName) {
    toolchain::SmallString<64> buf = UnitName;
    api().store_discard_unit(obj, buf.c_str());
  }

  void discardRecord(StringRef RecordName) {
    toolchain::SmallString<64> buf = RecordName;
    api().store_discard_record(obj, buf.c_str());
  }

  void getUnitNameFromOutputPath(StringRef outputPath, toolchain::SmallVectorImpl<char> &nameBuf) {
    toolchain::SmallString<256> buf = outputPath;
    toolchain::SmallString<64> unitName;
    unitName.resize(64);
    size_t nameLen = api().store_get_unit_name_from_output_path(obj, buf.c_str(), unitName.data(), unitName.size());
    if (nameLen+1 > unitName.size()) {
      unitName.resize(nameLen+1);
      api().store_get_unit_name_from_output_path(obj, buf.c_str(), unitName.data(), unitName.size());
    }
    nameBuf.append(unitName.begin(), unitName.begin()+nameLen);
  }

  toolchain::Optional<timespec>
  getUnitModificationTime(StringRef unitName, std::string &error) {
    toolchain::SmallString<64> buf = unitName;
    int64_t seconds, nanoseconds;
    indexstore_error_t c_err = nullptr;
    bool err = api().store_get_unit_modification_time(obj, buf.c_str(),
      &seconds, &nanoseconds, &c_err);
    if (err && c_err) {
      error = api().error_get_description(c_err);
      api().error_dispose(c_err);
      return toolchain::None;
    }
    timespec ts;
    ts.tv_sec = seconds;
    ts.tv_nsec = nanoseconds;
    return ts;
  }

  void purgeStaleData() {
    api().store_purge_stale_data(obj);
  }
};

class IndexRecordReader {
  indexstore_record_reader_t obj;
  IndexStoreLibraryRef lib;

public:
  IndexRecordReader(IndexStore &store, StringRef recordName, std::string &error) : lib(store.library) {
    toolchain::SmallString<64> buf = recordName;
    indexstore_error_t c_err = nullptr;
    obj = lib->api().record_reader_create(store.obj, buf.c_str(), &c_err);
    if (c_err) {
      error = lib->api().error_get_description(c_err);
      lib->api().error_dispose(c_err);
    }
  }

  IndexRecordReader(IndexRecordReader &&other) : obj(other.obj) {
    other.obj = nullptr;
  }

  ~IndexRecordReader() {
    lib->api().record_reader_dispose(obj);
  }

  bool isValid() const { return obj; }
  bool isInvalid() const { return !isValid(); }
  explicit operator bool() const { return isValid(); }

  /// Goes through and passes record decls, after filtering using a \c Checker
  /// function.
  ///
  /// Resulting decls can be used as filter for \c foreachOccurrence. This
  /// allows allocating memory only for the record decls that the caller is
  /// interested in.
  bool searchSymbols(toolchain::function_ref<bool(IndexRecordSymbol, bool &stop)> filter,
                     toolchain::function_ref<void(IndexRecordSymbol)> receiver) {
    auto forwarder_filter = [&](indexstore_symbol_t symbol, bool *stop) -> bool {
      return filter({symbol, lib}, *stop);
    };
    auto forwarder_receiver = [&](indexstore_symbol_t symbol) {
      receiver({symbol, lib});
    };
    return lib->api().record_reader_search_symbols_f(obj, &forwarder_filter, functionPtrFromFunctionRef<decltype(forwarder_filter)>,
                                                     &forwarder_receiver, functionPtrFromFunctionRef<decltype(forwarder_receiver)>);
  }

  bool foreachSymbol(bool noCache, toolchain::function_ref<bool(IndexRecordSymbol)> receiver) {
    auto forwarder = [&](indexstore_symbol_t sym) -> bool {
      return receiver({sym, lib});
    };
    return lib->api().record_reader_symbols_apply_f(obj, noCache, &forwarder, functionPtrFromFunctionRef<decltype(forwarder)>);
  }

  /// \param DeclsFilter if non-empty indicates the list of decls that we want
  /// to get occurrences for. An empty array indicates that we want occurrences
  /// for all decls.
  /// \param RelatedDeclsFilter Same as \c DeclsFilter but for related decls.
  bool foreachOccurrence(ArrayRef<IndexRecordSymbol> symbolsFilter,
                         ArrayRef<IndexRecordSymbol> relatedSymbolsFilter,
              toolchain::function_ref<bool(IndexRecordOccurrence)> receiver) {
    toolchain::SmallVector<indexstore_symbol_t, 16> c_symbolsFilter;
    c_symbolsFilter.reserve(symbolsFilter.size());
    for (IndexRecordSymbol sym : symbolsFilter) {
      c_symbolsFilter.push_back(sym.obj);
    }
    toolchain::SmallVector<indexstore_symbol_t, 16> c_relatedSymbolsFilter;
    c_relatedSymbolsFilter.reserve(relatedSymbolsFilter.size());
    for (IndexRecordSymbol sym : relatedSymbolsFilter) {
      c_relatedSymbolsFilter.push_back(sym.obj);
    }
    auto forwarder = [&](indexstore_occurrence_t occur) -> bool {
      return receiver({occur, lib});
    };
    return lib->api().record_reader_occurrences_of_symbols_apply_f(obj,
                                c_symbolsFilter.data(), c_symbolsFilter.size(),
                                c_relatedSymbolsFilter.data(),
                                c_relatedSymbolsFilter.size(),
                                &forwarder, functionPtrFromFunctionRef<decltype(forwarder)>);
  }

  bool foreachOccurrence(
              toolchain::function_ref<bool(IndexRecordOccurrence)> receiver) {
    auto forwarder = [&](indexstore_occurrence_t occur) -> bool {
      return receiver({occur, lib});
    };
    return lib->api().record_reader_occurrences_apply_f(obj, &forwarder, functionPtrFromFunctionRef<decltype(forwarder)>);
  }

  bool foreachOccurrenceInLineRange(unsigned lineStart, unsigned lineEnd,
              toolchain::function_ref<bool(IndexRecordOccurrence)> receiver) {
    auto forwarder = [&](indexstore_occurrence_t occur) -> bool {
      return receiver({occur, lib});
    };
    return lib->api().record_reader_occurrences_in_line_range_apply_f(obj,
                                                                      lineStart,
                                                                      lineEnd,
                                         &forwarder, functionPtrFromFunctionRef<decltype(forwarder)>);
  }
};

class IndexUnitDependency {
  indexstore_unit_dependency_t obj;
  IndexStoreLibraryRef lib;
  friend class IndexUnitReader;

public:
  IndexUnitDependency(indexstore_unit_dependency_t obj, IndexStoreLibraryRef lib) : obj(obj), lib(std::move(lib)) {}

  enum class DependencyKind {
    Unit,
    Record,
    File,
  };
  DependencyKind getKind() {
    switch (lib->api().unit_dependency_get_kind(obj)) {
    case INDEXSTORE_UNIT_DEPENDENCY_UNIT: return DependencyKind::Unit;
    case INDEXSTORE_UNIT_DEPENDENCY_RECORD: return DependencyKind::Record;
    case INDEXSTORE_UNIT_DEPENDENCY_FILE: return DependencyKind::File;
    }
  }
  bool isSystem() { return lib->api().unit_dependency_is_system(obj); }
  StringRef getName() { return stringFromIndexStoreStringRef(lib->api().unit_dependency_get_name(obj)); }
  StringRef getFilePath() { return stringFromIndexStoreStringRef(lib->api().unit_dependency_get_filepath(obj)); }
  StringRef getModuleName() { return stringFromIndexStoreStringRef(lib->api().unit_dependency_get_modulename(obj)); }

};

class IndexUnitInclude {
  indexstore_unit_include_t obj;
  IndexStoreLibraryRef lib;
  friend class IndexUnitReader;

public:
  IndexUnitInclude(indexstore_unit_include_t obj, IndexStoreLibraryRef lib) : obj(obj), lib(std::move(lib)) {}

  StringRef getSourcePath() {
    return stringFromIndexStoreStringRef(lib->api().unit_include_get_source_path(obj));
  }
  StringRef getTargetPath() {
    return stringFromIndexStoreStringRef(lib->api().unit_include_get_target_path(obj));
  }
  unsigned getSourceLine() {
    return lib->api().unit_include_get_source_line(obj);
  }
};

class IndexUnitReader {
  indexstore_unit_reader_t obj;
  IndexStoreLibraryRef lib;

public:
  IndexUnitReader(IndexStore &store, StringRef unitName, std::string &error) : lib(store.library) {
    toolchain::SmallString<64> buf = unitName;
    indexstore_error_t c_err = nullptr;
    obj = lib->api().unit_reader_create(store.obj, buf.c_str(), &c_err);
    if (c_err) {
      error = lib->api().error_get_description(c_err);
      lib->api().error_dispose(c_err);
    }
  }

  IndexUnitReader(IndexUnitReader &&other) : obj(other.obj) {
    other.obj = nullptr;
  }

  ~IndexUnitReader() {
    lib->api().unit_reader_dispose(obj);
  }

  bool isValid() const { return obj; }
  bool isInvalid() const { return !isValid(); }
  explicit operator bool() const { return isValid(); }

  StringRef getProviderIdentifier() {
    return stringFromIndexStoreStringRef(lib->api().unit_reader_get_provider_identifier(obj));
  }
  StringRef getProviderVersion() {
    return stringFromIndexStoreStringRef(lib->api().unit_reader_get_provider_version(obj));
  }

  timespec getModificationTime() {
    int64_t seconds, nanoseconds;
    lib->api().unit_reader_get_modification_time(obj, &seconds, &nanoseconds);
    timespec ts;
    ts.tv_sec = seconds;
    ts.tv_nsec = nanoseconds;
    return ts;
  }

  bool isSystemUnit() { return lib->api().unit_reader_is_system_unit(obj); }
  bool isModuleUnit() { return lib->api().unit_reader_is_module_unit(obj); }
  bool isDebugCompilation() { return lib->api().unit_reader_is_debug_compilation(obj); }
  bool hasMainFile() { return lib->api().unit_reader_has_main_file(obj); }

  StringRef getMainFilePath() {
    return stringFromIndexStoreStringRef(lib->api().unit_reader_get_main_file(obj));
  }
  StringRef getModuleName() {
    return stringFromIndexStoreStringRef(lib->api().unit_reader_get_module_name(obj));
  }
  StringRef getWorkingDirectory() {
    return stringFromIndexStoreStringRef(lib->api().unit_reader_get_working_dir(obj));
  }
  StringRef getOutputFile() {
    return stringFromIndexStoreStringRef(lib->api().unit_reader_get_output_file(obj));
  }
  StringRef getSysrootPath() {
    return stringFromIndexStoreStringRef(lib->api().unit_reader_get_sysroot_path(obj));
  }
  StringRef getTarget() {
    return stringFromIndexStoreStringRef(lib->api().unit_reader_get_target(obj));
  }

  bool foreachDependency(toolchain::function_ref<bool(IndexUnitDependency)> receiver) {
    auto forwarder = [&](indexstore_unit_dependency_t dep) -> bool {
      return receiver({dep, lib});
    };
    return lib->api().unit_reader_dependencies_apply_f(obj, &forwarder, functionPtrFromFunctionRef<decltype(forwarder)>);;
  }

  bool foreachInclude(toolchain::function_ref<bool(IndexUnitInclude)> receiver) {
    auto forwarder = [&](indexstore_unit_include_t inc) -> bool {
      return receiver({inc, lib});
    };
    return lib->api().unit_reader_includes_apply_f(obj, &forwarder, functionPtrFromFunctionRef<decltype(forwarder)>);
  }
};

} // namespace indexstore

#endif
