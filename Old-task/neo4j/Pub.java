package org.neo4j.annotations.api;

import java.util.HashSet;
import java.util.List;
import java.util.Set;
import javax.lang.model.element.Element;
import javax.lang.model.element.PackageElement;
import javax.lang.model.element.QualifiedNameable;
import jdk.javadoc.doclet.DocletEnvironment;
import jdk.javadoc.doclet.StandardDoclet;
import jdk.javadoc.internal.tool.DocEnvImpl;

public class PublicApiDoclet extends StandardDoclet {
    @Override
    public String getName() {
        return "PublicApiDoclet";
    }

    @Override
    public boolean run(DocletEnvironment docEnv) {
        FilteringDocletEnvironment docletEnvironment = new FilteringDocletEnvironment(docEnv);
        return super.run(docletEnvironment);
    }

    private static class FilteringDocletEnvironment extends DocEnvImpl {
        private final DocletEnvironment docEnv;

        FilteringDocletEnvironment(DocletEnvironment docEnv) {
            super(((DocEnvImpl) docEnv).toolEnv, ((DocEnvImpl) docEnv).etable);
            this.docEnv = docEnv;
        }

        @Override
        public Set<? extends Element> getIncludedElements() {
            Set<Element> includedElements = new HashSet<>(docEnv.getIncludedElements());
            includedElements.removeIf(element -> !includeElement(element));
            return includedElements;
        }

        @Override
        public boolean isIncluded(Element e) {
            if (e instanceof QualifiedNameable) {
                return includeElement(e);
            }
            return super.isIncluded(e);
        }

        @Override
        public boolean isSelected(Element e) {
            if (e instanceof QualifiedNameable) {
                return includeElement(e);
            }
            return super.isIncluded(e);
        }

        private boolean includeElement(Element element) {
            if (element.getAnnotation(PublicApi.class) != null) {
                return true;
            }
            Element enclosingElement = element.getEnclosingElement();
            if (enclosingElement != null && enclosingElement.getAnnotation(PublicApi.class) != null) {
                return true;
            }
            if (element instanceof PackageElement) {
                return includePackage((PackageElement) element);
            }
            return false;
        }

        private boolean includePackage(PackageElement packageElement) {
            List<? extends Element> enclosedElements = packageElement.getEnclosedElements();
            for (Element enclosedElement : enclosedElements) {
                if (includeElement(enclosedElement)) {
                    return true;
                }
            }
            return false;
        }
    }
}