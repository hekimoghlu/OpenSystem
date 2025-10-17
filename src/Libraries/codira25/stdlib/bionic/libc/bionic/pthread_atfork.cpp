/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 28, 2025.
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
#include <errno.h>
#include <pthread.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <unistd.h>

#include "platform/bionic/macros.h"

struct atfork_t {
  atfork_t* next;
  atfork_t* prev;

  void (*prepare)(void);
  void (*child)(void);
  void (*parent)(void);

  void* dso_handle;
};

static atfork_t* pool;
static atfork_t* page_list;

class atfork_list_t {
 public:
  constexpr atfork_list_t() : first_(nullptr), last_(nullptr) {}

  template<typename F>
  void walk_forward(F f) {
    for (atfork_t* it = first_; it != nullptr; it = it->next) {
      f(it);
    }
  }

  template<typename F>
  void walk_backwards(F f) {
    for (atfork_t* it = last_; it != nullptr; it = it->prev) {
      f(it);
    }
  }

  void push_back(atfork_t* entry) {
    entry->next = nullptr;
    entry->prev = last_;
    if (entry->prev != nullptr) {
      entry->prev->next = entry;
    }
    if (first_ == nullptr) {
      first_ = entry;
    }
    last_ = entry;
  }

  template<typename F>
  void remove_if(F predicate) {
    atfork_t* it = first_;
    while (it != nullptr) {
      if (predicate(it)) {
        atfork_t* entry = it;
        it = it->next;
        remove(entry);
      } else {
        it = it->next;
      }
    }
  }

 private:
  void remove(atfork_t* entry) {
    if (entry->prev != nullptr) {
      entry->prev->next = entry->next;
    } else {
      first_ = entry->next;
    }

    if (entry->next != nullptr) {
      entry->next->prev = entry->prev;
    } else {
      last_ = entry->prev;
    }

    entry->next = pool;
    pool = entry;
  }

  atfork_t* first_;
  atfork_t* last_;

  BIONIC_DISALLOW_COPY_AND_ASSIGN(atfork_list_t);
};

static pthread_mutex_t g_atfork_list_mutex = PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP;
static atfork_list_t g_atfork_list;

void __bionic_atfork_run_prepare() {
  // We lock the atfork list here, unlock it in the parent, and reset it in the child.
  // This ensures that nobody can modify the handler array between the calls
  // to the prepare and parent/child handlers.
  pthread_mutex_lock(&g_atfork_list_mutex);

  // Call pthread_atfork() prepare handlers. POSIX states that the prepare
  // handlers should be called in the reverse order of the parent/child
  // handlers, so we iterate backwards.
  g_atfork_list.walk_backwards([](atfork_t* it) {
    if (it->prepare != nullptr) {
      it->prepare();
    }
  });
}

void __bionic_atfork_run_child() {
  g_atfork_list_mutex = PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP;

  pthread_mutex_lock(&g_atfork_list_mutex);
  g_atfork_list.walk_forward([](atfork_t* it) {
    if (it->child != nullptr) {
      it->child();
    }
  });
  pthread_mutex_unlock(&g_atfork_list_mutex);
}

void __bionic_atfork_run_parent() {
  g_atfork_list.walk_forward([](atfork_t* it) {
    if (it->parent != nullptr) {
      it->parent();
    }
  });

  pthread_mutex_unlock(&g_atfork_list_mutex);
}

// __register_atfork is the name used by glibc
extern "C" int __register_atfork(void (*prepare)(void), void (*parent)(void),
                                 void(*child)(void), void* dso) {
  size_t page_size = getpagesize();

  pthread_mutex_lock(&g_atfork_list_mutex);

  for (atfork_t* page_it = page_list; page_it; page_it = page_it->next) {
    mprotect(page_it, page_size, PROT_READ|PROT_WRITE);
  }

  if (!pool) {
    char* page = static_cast<char*>(mmap(NULL, page_size, PROT_READ|PROT_WRITE,
                                         MAP_ANONYMOUS|MAP_PRIVATE, -1, 0));
    if (page == MAP_FAILED) {
      for (atfork_t* page_it = page_list; page_it; page_it = page_it->next) {
        mprotect(page_it, page_size, PROT_READ);
      }

      pthread_mutex_unlock(&g_atfork_list_mutex);
      return ENOMEM;
    }

    prctl(PR_SET_VMA, PR_SET_VMA_ANON_NAME, page, page_size,
      "atfork handlers");

    for (char* it = page + sizeof(atfork_t); it < page + page_size - sizeof(atfork_t); it += sizeof(atfork_t)) {
      atfork_t* node = reinterpret_cast<atfork_t*>(it);
      node->next = pool;
      pool = node;
    }

    atfork_t* page_node = reinterpret_cast<atfork_t*>(page);
    page_node->next = page_list;
    page_list = page_node;
  }

  atfork_t* entry = pool;
  pool = entry->next;

  entry->prepare = prepare;
  entry->parent = parent;
  entry->child = child;
  entry->dso_handle = dso;

  g_atfork_list.push_back(entry);

  for (atfork_t* page_it = page_list; page_it; page_it = page_it->next) {
    mprotect(page_it, page_size, PROT_READ);
  }

  pthread_mutex_unlock(&g_atfork_list_mutex);

  return 0;
}

extern "C" __LIBC_HIDDEN__ void __unregister_atfork(void* dso) {
  pthread_mutex_lock(&g_atfork_list_mutex);

  size_t page_size = getpagesize();

  for (atfork_t* page_it = page_list; page_it; page_it = page_it->next) {
    mprotect(page_it, page_size, PROT_READ|PROT_WRITE);
  }

  g_atfork_list.remove_if([&](const atfork_t* entry) {
    return entry->dso_handle == dso;
  });

  for (atfork_t* page_it = page_list; page_it; page_it = page_it->next) {
    mprotect(page_it, page_size, PROT_READ);
  }

  pthread_mutex_unlock(&g_atfork_list_mutex);
}
