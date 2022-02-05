# 兼容 vue

## 基础用法

1. 在`index.html`文件中引入

```js
<script src="//cdn.jsdelivr.net/npm/vue"></script>
<script src="//cdn.jsdelivr.net/npm/docsify"></script>

<!-- 压缩版 -->
<script src="//cdn.jsdelivr.net/npm/vue/dist/vue.min.js"></script>
<script src="//cdn.jsdelivr.net/npm/docsify/lib/docsify.min.js"></script>
```

2. 直接写入接口使用

```html
<ul>
  <li v-for="i in 10">{{ i }}</li>
</ul>
```

<ul>
  <li v-for="i in 10">{{ i }}</li>
</ul>

## 结合 vuep 渲染 vue 组件

```html
<!-- Inject CSS file -->
<link rel="stylesheet" href="//cdn.jsdelivr.net/npm/vuep/dist/vuep.css" />

<!-- Inject JavaScript file -->
<script src="//cdn.jsdelivr.net/npm/vue"></script>
<script src="//cdn.jsdelivr.net/npm/vuep"></script>
<script src="//cdn.jsdelivr.net/npm/docsify"></script>

<!-- 压缩版 -->
<script src="//cdn.jsdelivr.net/npm/vue/dist/vue.min.js"></script>
<script src="//cdn.jsdelivr.net/npm/vuep/dist/vuep.min.js"></script>
<script src="//cdn.jsdelivr.net/npm/docsify/lib/docsify.min.js"></script>
```

<vuep template="#example"></vuep>

<script v-pre type="text/x-template" id="example">
  <template>
    <div>Hello, {{ name }}!</div>
  </template>

  <script>
    module.exports = {
      data: function () {
        return { name: 'Vue' }
      }
    }
  </script>
</script>
