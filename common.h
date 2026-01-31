#ifndef INTERACTION_SAW_MODELS_COMMON_H
#define INTERACTION_SAW_MODELS_COMMON_H

#include <Kokkos_Core.hpp>
#include <string>
#include <type_traits>

 

template<class DstView, class SrcView>
void realloc_like_and_copy(DstView& dst, const SrcView& src, const std::string& label) {
  static_assert(int(DstView::rank) == int(SrcView::rank), "rank mismatch");
  static_assert(std::is_same_v<typename DstView::value_type, typename SrcView::value_type>,
                "value_type mismatch");

  // Allocate/resize dst to match src
  if constexpr (DstView::rank == 0) {
    // scalar View: constructor takes only label
    if (!dst.is_allocated()) 
      dst = DstView(label);
  } else if constexpr (DstView::rank == 1) {
    if (!dst.is_allocated() || dst.extent(0) != src.extent(0)) {
      dst = DstView(label, src.extent(0));
    }
  } else if constexpr (DstView::rank == 2) {
    if (!dst.is_allocated() || dst.extent(0) != src.extent(0) || dst.extent(1) != src.extent(1)) {
      dst = DstView(label, src.extent(0), src.extent(1));
    }
  } else if constexpr (DstView::rank == 3) {
    if (!dst.is_allocated() || dst.extent(0) != src.extent(0) || dst.extent(1) != src.extent(1) ||
        dst.extent(2) != src.extent(2)) {
      dst = DstView(label, src.extent(0), src.extent(1), src.extent(2));
    }
  } else {
    static_assert(DstView::rank <= 3, "extend realloc_like_and_copy for higher rank");
  }

  Kokkos::deep_copy(dst, src);
}


#endif