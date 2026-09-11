#include "nedo/model.hpp"
#include "nedo/kernels.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include <pybind11/functional.h>
namespace py=pybind11;
PYBIND11_MODULE(_core,m){
    m.doc()="NedoLM native GGUF runtime core";
    m.def("set_device",[](const std::string& device){ nedo::kernels::set_device(device); });
    m.def("device",[](){ return nedo::kernels::device(); });
    m.def("cuda_available",[](){ return nedo::kernels::cuda_available(); });
    m.def("cuda_device_name",[](){ return nedo::kernels::cuda_device_name(); });
    py::class_<nedo::GenerationConfig>(m,"GenerationConfig").def(py::init<>()).def_readwrite("max_new_tokens",&nedo::GenerationConfig::max_new_tokens).def_readwrite("temperature",&nedo::GenerationConfig::temperature).def_readwrite("top_p",&nedo::GenerationConfig::top_p).def_readwrite("top_k",&nedo::GenerationConfig::top_k).def_readwrite("seed",&nedo::GenerationConfig::seed).def_readwrite("add_bos",&nedo::GenerationConfig::add_bos);
    py::class_<nedo::ModelConfig>(m,"ModelConfig").def_readonly("vocab",&nedo::ModelConfig::vocab).def_readonly("dim",&nedo::ModelConfig::dim).def_readonly("layers",&nedo::ModelConfig::layers).def_readonly("heads",&nedo::ModelConfig::heads).def_readonly("kv_heads",&nedo::ModelConfig::kv_heads).def_readonly("head_dim",&nedo::ModelConfig::head_dim).def_readonly("ffn",&nedo::ModelConfig::ffn).def_readonly("context",&nedo::ModelConfig::context).def_readonly("sliding",&nedo::ModelConfig::sliding);
    py::class_<nedo::SchemaReport>(m,"SchemaReport").def_readonly("architecture_ok",&nedo::SchemaReport::architecture_ok).def_readonly("standard_tensors_ok",&nedo::SchemaReport::standard_tensors_ok).def_readonly("morph_tensors",&nedo::SchemaReport::morph_tensors).def_readonly("morph_names",&nedo::SchemaReport::morph_names).def_readonly("missing",&nedo::SchemaReport::missing).def_readonly("unmatched",&nedo::SchemaReport::unmatched).def_readonly("router_contract",&nedo::SchemaReport::router_contract);
    py::class_<nedo::NedoModel,std::shared_ptr<nedo::NedoModel>>(m,"Model").def(py::init<const std::filesystem::path&>()).def_property_readonly("config",&nedo::NedoModel::config,py::return_value_policy::reference_internal).def_property_readonly("schema",&nedo::NedoModel::schema,py::return_value_policy::reference_internal).def("summary",&nedo::NedoModel::summary).def("tokenize",&nedo::NedoModel::tokenize,py::arg("text"),py::arg("add_bos")=false).def("detokenize",&nedo::NedoModel::detokenize).def("generate_ids",[](nedo::NedoModel& self,const std::vector<uint32_t>& ids,const nedo::GenerationConfig& cfg){ return self.generate_ids(ids,cfg); },py::arg("prompt_ids"),py::arg("config")=nedo::GenerationConfig{}).def("generate_ids_stream",[](nedo::NedoModel& self,const std::vector<uint32_t>& ids,const nedo::GenerationConfig& cfg,py::function callback){ return self.generate_ids(ids,cfg,[callback](uint32_t id){ callback(id); }); },py::arg("prompt_ids"),py::arg("config"),py::arg("callback")).def("generate",&nedo::NedoModel::generate,py::arg("prompt"),py::arg("config")=nedo::GenerationConfig{});
}
